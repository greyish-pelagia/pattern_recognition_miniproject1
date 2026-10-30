import ast
import importlib
import importlib.util
from pathlib import Path
import re
import sys
import types
import unittest
import weakref
from unittest.mock import patch

from PIL import Image

import src.enhancement as enhancement


class RestorationTests(unittest.TestCase):
    def test_empty_input_needs_no_model(self):
        self.assertEqual(enhancement.restore_images([]), [])

    def test_invalid_images_fail_before_model_loading(self):
        for value in (None, Image.new('RGB', (4, 4)), ['image.png']):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(TypeError):
                    enhancement.restore_images(value)

    def test_restores_mixed_sizes_and_modes_in_order_without_mutating_inputs(self):
        pipeline = importlib.import_module('src.enhancement.pipeline')
        inputs = [Image.new('L', (71, 95), 30), Image.new('RGBA', (129, 64), (90, 40, 20, 255))]
        original_bytes = [image.tobytes() for image in inputs]

        class Tensor:
            def __init__(self, image):
                self.image = image

            def unsqueeze(self, axis):
                return self

            def to(self, device):
                self_test.assertEqual(str(device), 'mps')
                return self

        class Model:
            def parameters(self):
                return iter([types.SimpleNamespace(device='mps')])

            def batchify_sample(self, tensor, captions, **settings):
                self_test.assertEqual(captions, [''])
                self_test.assertEqual(settings['num_samples'], 1)
                self_test.assertEqual(tensor.image.mode, 'RGB')
                return [tensor]

        def prepare(image):
            return Tensor(image), image.height, image.width

        def finish(tensor, height, width):
            return tensor.image.copy().resize((width, height))

        self_test = self
        with patch.object(pipeline, '_load_runtime', return_value=(Model(), prepare, finish)):
            outputs = enhancement.restore_images(inputs, device='mps')
        self.assertEqual([image.size for image in outputs], [image.size for image in inputs])
        self.assertEqual([image.mode for image in outputs], ['RGB', 'RGB'])
        self.assertEqual([image.getpixel((0, 0)) for image in outputs], [(30, 30, 30), (90, 40, 20)])
        self.assertEqual([image.tobytes() for image in inputs], original_bytes)
        self.assertTrue(all(out is not inp for out, inp in zip(outputs, inputs)))

    def test_missing_checkpoint_configuration_is_actionable(self):
        with self.assertRaisesRegex(ValueError, 'sdxl_checkpoint.*supir_checkpoint'):
            enhancement.restore_images([Image.new('RGB', (64, 64))])

    def test_missing_checkpoint_file_is_reported_before_loading_torch(self):
        with self.assertRaises(FileNotFoundError):
            enhancement.restore_images(
                [Image.new('RGB', (64, 64))],
                sdxl_checkpoint='/missing/sdxl.safetensors',
                supir_checkpoint='/missing/supir.ckpt',
            )

    def test_vendored_targets_resolve_without_top_level_module_aliases(self):
        # Only configuration/import resolution is exercised, not tensor operations.
        name = 'src.enhancement.sgm.util'
        with patch.dict(sys.modules, {'torch': types.ModuleType('torch')}):
            try:
                util = importlib.import_module(name)
                self.assertIs(util.get_obj_from_str('sgm.util.default'), util.default)
                self.assertIs(util.get_obj_from_str('PIL.Image.Image'), Image.Image)
            finally:
                sys.modules.pop(name, None)

    def test_checkpoint_switch_releases_old_model_before_loading_replacement(self):
        pipeline = importlib.import_module('src.enhancement.pipeline')
        self.assertTrue(hasattr(pipeline, '_build_runtime'))
        references = []

        class Model:
            pass

        def build(*args):
            if references:
                self.assertIsNone(references[-1](), 'Old model still occupies GPU memory')
            model = Model()
            references.append(weakref.ref(model))
            return model, None, None

        with patch.object(pipeline, '_RUNTIME', None), patch.object(pipeline, '_RUNTIME_KEY', None):
            with patch.object(pipeline, '_build_runtime', side_effect=build) as factory:
                pipeline._load_runtime('sdxl', 'Q', None, None)
                pipeline._load_runtime('sdxl', 'Q', None, None)
                self.assertEqual(factory.call_count, 1)
                pipeline._load_runtime('sdxl', 'F', None, None)
                self.assertEqual(factory.call_count, 2)

    def test_device_switch_invalidates_cached_runtime(self):
        pipeline = importlib.import_module('src.enhancement.pipeline')
        with patch.object(pipeline, '_RUNTIME', None), patch.object(pipeline, '_RUNTIME_KEY', None):
            with patch.object(pipeline, '_build_runtime', return_value=(object(), None, None)) as factory:
                pipeline._load_runtime('sdxl', 'Q', None, None, 'mps')
                pipeline._load_runtime('sdxl', 'Q', None, None, 'cuda:0')
                self.assertEqual(factory.call_count, 2)

    def test_automatically_selects_mps_when_cuda_is_unavailable(self):
        pipeline = importlib.import_module('src.enhancement.pipeline')
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
            device=lambda value: types.SimpleNamespace(type=value, __str__=lambda: value),
        )
        self.assertEqual(pipeline._select_device(torch, None).type, 'mps')

    def test_explicit_unavailable_mps_has_actionable_error(self):
        pipeline = importlib.import_module('src.enhancement.pipeline')
        torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
            device=lambda value: types.SimpleNamespace(type=value),
        )
        with self.assertRaisesRegex(RuntimeError, 'MPS.*not available'):
            pipeline._select_device(torch, 'mps')


class DependencyTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / 'src' / 'enhancement'

    def test_all_relative_imports_reference_retained_symbols(self):
        for path in self.root.rglob('*.py'):
            package = 'src.enhancement' + '.' + '.'.join(path.parent.relative_to(self.root).parts)
            package = package.rstrip('.')
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom) or not node.level:
                    continue
                module = importlib.util.resolve_name('.' * node.level + (node.module or ''), package)
                target = self._module_path(module)
                self.assertTrue(target.exists(), f'{path}: missing {module}')
                tree = ast.parse(target.read_text())
                names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
                for item in tree.body:
                    if isinstance(item, (ast.Import, ast.ImportFrom)):
                        names.update(a.asname or a.name for a in item.names)
                    elif isinstance(item, ast.Assign):
                        names.update(n.id for n in item.targets if isinstance(n, ast.Name))
                for alias in node.names:
                    self.assertIn(alias.name, names, f'{path}: missing {module}.{alias.name}')

    def _module_path(self, module):
        target = self.root.joinpath(*module.removeprefix('src.enhancement.').split('.'))
        return target.with_suffix('.py') if target.with_suffix('.py').exists() else target / '__init__.py'

    def test_yaml_and_dynamic_targets_reference_retained_definitions(self):
        texts = [p.read_text() for p in self.root.rglob('*.yaml')]
        texts.extend(p.read_text() for p in self.root.rglob('*.py'))
        text = '\n'.join(texts)
        pattern = r'(?:SUPIR|sgm)(?:\.[A-Za-z_]\w*)+'
        targets = set(re.findall(r'(?:target:|network_wrapper:)\s*(' + pattern + r')', text))
        targets.update(re.findall(r'[\x22\x27](' + pattern + r')[\x22\x27]', text))
        for target in targets:
            module, name = target.rsplit('.', 1)
            path = self._module_path('src.enhancement.' + module)
            self.assertTrue(path.exists(), target)
            tree = ast.parse(path.read_text())
            names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            names.update(a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names)
            self.assertIn(name, names, target)


if __name__ == '__main__':
    unittest.main()

import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { parseSync } from '@slidev/parser'

const source = readFileSync('slides.md', 'utf8')
const deck = parseSync(source, 'slides.md')
assert.equal(deck.slides.length, 12, 'Prezentacja powinna mieć 12 slajdów')

let seconds = 0
let words = 0
for (const slide of deck.slides) {
  assert.ok(slide.content.trim(), 'Pusty slajd')
  const timing = slide.note?.match(/^Czas: (\d+) s\./)
  assert.ok(timing, `Brak czasu i notatek: ${slide.title}`)
  seconds += Number(timing[1])
  const speech = slide.note.split('\n\n')[1]
  words += speech.trim().split(/\s+/).length
}
assert.equal(seconds, 460, 'Planowany czas: 7 minut 40 sekund')

const provenance = JSON.parse(readFileSync('sources.json', 'utf8'))
for (const [, file, alt] of source.matchAll(/<img[^>]*src="\/images\/([^"]+)"[^>]*alt="([^"]+)"/g)) {
  assert.ok(existsSync(`public/images/${file}`), `Brak obrazu: ${file}`)
  assert.ok(alt.trim(), `Brak opisu obrazu: ${file}`)
  assert.ok(provenance.images[file], `Brak źródła obrazu: ${file}`)
}

console.log(`${deck.slides.length} slajdów; ${seconds} s; ${words} słów w notatkach; obrazy i źródła OK`)

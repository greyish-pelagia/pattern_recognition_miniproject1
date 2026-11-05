<script setup>
const models = [
  { name: 'SVM · RBF', simple: 91 / 93, hard: 580 / 600 },
  { name: 'SVM · liniowy', simple: 91 / 93, hard: 578 / 600 },
  { name: 'Regresja logistyczna', simple: 90 / 93, hard: 581 / 600 },
  { name: 'Random Forest', simple: 90 / 93, hard: 569 / 600 },
  { name: 'kNN', simple: 88 / 93, hard: 560 / 600 },
]

const format = new Intl.NumberFormat('pl-PL', {
  style: 'percent', maximumFractionDigits: 1, minimumFractionDigits: 1,
})
</script>

<template>
  <table class="results-chart">
    <caption>Accuracy testowa · skala wykresu 90–100%</caption>
    <thead>
      <tr><th>Model</th><th>Zadanie proste</th><th>Zadanie trudne</th></tr>
    </thead>
    <tbody>
      <tr v-for="model in models" :key="model.name">
        <th>{{ model.name }}</th>
        <td v-for="task in ['simple', 'hard']" :key="task">
          <div class="result-track">
            <span class="result-dot" :style="{ left: `${(model[task] - 0.9) * 1000}%` }"></span>
          </div>
          <strong>{{ format.format(model[task]) }}</strong>
        </td>
      </tr>
    </tbody>
  </table>
</template>

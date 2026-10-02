<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { queryShopTypeList, queryNearShop } from '../api'

const router = useRouter()
const types = ref([])
const form = reactive({ typeId: 1, x: 116.31, y: 39.99, distKm: 20 })
const result = ref(null)

onMounted(async () => {
  const r = await queryShopTypeList()
  types.value = r.data
})

async function search() {
  const r = await queryNearShop(form.typeId, form.x, form.y, form.distKm)
  result.value = r.data
}
</script>

<template>
  <div class="page">
    <div class="page-title">附近优惠</div>
    <div class="page-sub">按你的位置,推荐半径内的优惠商铺</div>
    <div class="card">
      <h3>位置搜索</h3>
      <p class="muted" style="margin:8px 0">输入经纬度和搜索半径,查看附近商铺</p>
      <div class="row" style="margin-top:12px">
        <span>类型:</span>
        <el-select v-model="form.typeId" style="width:120px">
          <el-option v-for="t in types" :key="t.id" :label="t.name" :value="t.id" />
        </el-select>
        <span>经度:</span>
        <el-input-number v-model="form.x" :precision="6" :step="0.01" />
        <span>纬度:</span>
        <el-input-number v-model="form.y" :precision="6" :step="0.01" />
        <span>半径(km):</span>
        <el-input-number v-model="form.distKm" :min="1" :max="100" />
        <el-button type="primary" @click="search">搜索</el-button>
      </div>
    </div>

    <div class="card" v-if="result">
      <div class="card-title">
        <h3>附近商铺</h3>
        <span class="muted">共 {{ result.shops.length }} 家</span>
      </div>
      <div v-for="s in result.shops" :key="s.id" class="near-item" @click="router.push(`/shop/${s.id}`)">
        <img :src="s.cover || `/covers/shop_${s.id}.svg`" class="near-cover" />
        <div class="near-body">
          <h4>{{ s.name }}</h4>
          <div class="near-meta">
            <span class="rate">★ {{ (s.score / 20).toFixed(1) }}</span>
            <span class="amount">¥{{ (s.avgPrice / 100).toFixed(0) }}/人</span>
            <span class="muted">销量 {{ s.sold }}</span>
          </div>
          <p class="muted">{{ s.address }}</p>
        </div>
        <div class="near-dist">
          <div class="dist-num">{{ result.distances[s.id]?.toFixed(1) }}</div>
          <div class="dist-unit">km</div>
        </div>
      </div>
      <el-empty v-if="!result.shops.length" description="半径内暂无商铺,试试加大距离" />
    </div>
  </div>
</template>

<style scoped>
.near-item { display: flex; align-items: center; gap: 14px; padding: 12px 0; border-bottom: 1px solid var(--line); cursor: pointer; }
.near-item:hover { background: var(--brand-light); }
.near-item:last-child { border-bottom: none; }
.near-cover { width: 90px; height: 70px; border-radius: 8px; object-fit: cover; }
.near-body { flex: 1; }
.near-body h4 { font-size: 15px; font-weight: 600; }
.near-meta { display: flex; align-items: center; gap: 10px; font-size: 12px; margin: 4px 0; }
.near-meta .rate { color: var(--brand); font-weight: 600; }
.near-dist { text-align: center; min-width: 56px; }
.dist-num { font-size: 20px; font-weight: 700; color: var(--brand); }
.dist-unit { font-size: 11px; color: var(--ink-3); }
</style>

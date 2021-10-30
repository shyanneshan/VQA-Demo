// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import store from './store'

import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import axios from 'axios'
import VCharts from 'v-charts'

import echarts from 'echarts'
Vue.prototype.$echarts = echarts
Vue.prototype.$axios = axios;
axios.defaults.baseURL = '/api';
axios.defaults.timeout=360000
// axios.defaults.baseURL = '/';
axios.interceptors.request.use(
  config => {
    const token = store.state.token;
    if (token) {
      // config.headers.Authorization = token;
      // config.headers["token"] = token;
    }
    return config;
  },
  error => {
    console.log("[main.js -> axios] Error! ");
    return Promise.reject(error);
  }
);
axios.interceptors.response.use(
  res => {
    return res;
  },
  error => {
    if (error.response) {
      switch (error.response.status) {
        case 403:
          this.$store.commit('clearUserInfo');
          router.replace({
            path: '/',
            query: {
              redirect: router.currentRoute.fullPath
            }
          });
      }
    }
    return Promise.reject(error.response.data);
  }
);

Vue.config.productionTip = false;

Vue.use(ElementUI);
Vue.use(VCharts);

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  store,
  // render: h => h(App),
  components: { App },
  template: '<App/>'
});
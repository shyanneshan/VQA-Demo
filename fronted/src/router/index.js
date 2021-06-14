import Vue from 'vue'
import Router from 'vue-router'

import vqa from "../pages/vqa";
import consult2 from "../components/Vqa/AI";
import dataset from "../components/Vqa/dataset";
import modelVqa from "../components/Vqa/ModelEvaluation";
import report from "../components/Vqa/report";
import label from "../components/Vqa/label";



Vue.use(Router);

const router = new Router({
  // 使用 history 模式消除 URL 中的 # 号
  // mode: "history",
  linkActiveClass: 'is-active',
  routes: [

    {
      path: '/vqa',
      // redirect:'/vqa/dataset',
      component: vqa,
      children: [
        {
          path: '',
          redirect: 'dataset'
        }, {
          path: 'dataset',
          name: 'dataset',
          component: dataset,
          meta: {
            keepAlive: true
          }
        }, {
          path: 'label',
          name: 'label',
          component: label,
          meta: {
            keepAlive: true
          }
        },
        {
          path: 'modelEvaluation',
          name: 'modelVqa',
          component: modelVqa
        }, {
          path: 'report',
          name: 'report',
          component: report
        }, {
          path: 'AI',
          name: 'AI',
          component: consult2
        }
      ]
    },
    {
        path: '/',
        redirect: '/vqa'
      }
  ]
});

// 导航守卫
// 使用 router.beforeEach 注册一个全局前置守卫，判断用户是否登陆
// router.beforeEach((to, from, next) => {
//   if (to.path === '/login') {
//     // alert("可添加“路由至登录页”的效果！！记得弄一下！！");
//     next();
//   } else if (to.path === '/signUp') {
//     next();
//   } else {
//     // Why I write this??
//     // let token = localStorage.getItem('Authorization');

//     let userInfo = JSON.parse(sessionStorage.getItem("addsCurrentUserInfo"));
//     let token = sessionStorage.getItem("addsCurrentUserToken");
//     if (token === 'null' || token === '' || userInfo === null || userInfo === '') {
//       next('/login');
//     } else {
//       next();
//     }
//   }
// });

export default router;

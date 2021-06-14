<template>
  <el-header height="80px">
    <div>
      <!-- <div class="nav-left">
        <img src="../assets/logo.jpg" height="79" alt="LOGO" @click="goToMainPage">
      </div> -->
      <div class="nav-center">
        <ul>
          <li v-for="(list) in headerList" :key="list.id">
            <router-link :to="{path: list.path}" onmouseover="this.style.color = '#409EFF'" onmouseout="this.style.color = 'rgb(140, 197, 255)'">
              {{list.title}}
            </router-link>
          </li>
        </ul>
      </div>
<!--      <div class="nav-right">-->
<!--        <span>{{'Hi, ' + this.$store.state.user.username}}</span>&nbsp;|&nbsp;<el-button type="text" @click="logOut">Log Out</el-button>-->
<!--      </div>-->
    </div>
  </el-header>
</template>

<script>
    export default {
        name: "NavigationBar",
        data() {
            return {
                headerList: [
                    // {id: '1', path: '/deepLearning', title: 'Deep Learning'},
                    // {id: '2', path: '/medicalArchivePreprocess', title: 'Medical Archive Preprocess'},
                    // {id: '3', path: '/knowledgeGraph', title: 'Knowledge Graph'},
                    // {id: '4', path: '/autoDiagnosis', title: 'Automatic Diagnosis'},
                    // {id: '5', path: '/QA', title: 'Q&A'},
                    // {id: '6', path: '/consult', title: 'Consult'},
                    // {id: '7', path: '/my', title: 'My'},
                    {id: '8', path: '/vqa', title: 'Vqa'}
                ],
                isShow: false
            };
        },
        methods: {
            goToMainPage() {
                this.$router.push('/deepLearning');
            },
            logOut() {
                this.$confirm('Log Out? ', 'Notification', {
                    confirmButtonText: 'Confirm',
                    cancelButtonText: 'Cancel',
                    type: 'warning'
                }).then(() => {
                    window.location.reload();
                    this.$store.commit('clearUserInfo');
                    this.$router.push('/');
                    this.$notify({
                        title: 'Success',
                        message: 'Log out successfully! ',
                        type: 'success'
                    });
                }).catch(() => {
                });
            }
        }
    }
</script>

<style scoped>
  .el-header {
    position: fixed;
    top: 0;

    /*left: 30px;*/
    /*right: 30px;*/

    left: 0;
    right: 0;

    border-bottom: 1px solid #dcdfe6;
    padding-left: 30px;
    /*color: #333;*/
    /*text-align: center;*/
    font-size: 16px;
    background-color: white;
    z-index: 999;
  }

  ul {
    line-height: 76px;
    margin: 0;
    padding-left: 0;
    /*padding-left: 20px;*/
  }

  li {
    list-style: none;
    float: left;
    margin-left: 40px;
    /*height: 80px;*/
  }

  a {
    text-decoration: none;
    display: block;
    /* padding: 0 10px; */
    color: rgb(140, 197, 255);
    /*font-weight: bolder;*/
  }

  .nav-left {
    height: 80px;
    float: left;
    cursor: pointer;
  }

  .nav-center {
    float: left;
    /*margin-left: 50px;*/
  }

  .nav-center ul li a.is-active {
    color: #409EFF;
    border-bottom: 4px solid rgb(140, 197, 255);
  }

  .nav-right {
    float: right;
    line-height: 80px;
  }
</style>

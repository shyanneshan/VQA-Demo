<template>
  <div id="app">
    <keep-alive :include="includedComponents" include="">
      <router-view/>
    </keep-alive>
  </div>
</template>

<script>
export default {
    name: 'App',
    data() {
        return {
            // includedComponents: [
            //     "Questions",
            //     "DeepModels",
            //     "knowledgeGraph",
            // ]
        };
    }
}
</script>

<style>
  body {
    margin: 0 30px;
  }

  #app {
    font-family: 'Avenir', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    /*text-align: center;*/
    /*color: #2c3e50;*/
    /*margin-top: 60px;*/
  }

  .scrollable-div .el-scrollbar {
    height: 100%;
  }

  .scrollable-div .el-scrollbar__wrap {
    overflow-x: auto;
  }

  ::-webkit-scrollbar {
    width: 2px;
  }
  ::-webkit-scrollbar-thumb {
    border-radius: 2px;
    height: 20px;
    background: #aaaaaa;
    /*background: rgba(0,0,0,0.4);*/
  }
  ::-webkit-scrollbar-track {
    /*box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.2);*/
    border-radius: 2px;
    /*background: rgba(0,0,0,0.4);*/
    /*background: #eee;*/
  }

  .el-tooltip__popper {
    max-width: 300px;
  }

  .el-upload--text {
    display: inline-block;
    vertical-align: top;
  }

  .el-upload-list {
    display: inline-block;
    margin-left: 10px;
  }
</style>

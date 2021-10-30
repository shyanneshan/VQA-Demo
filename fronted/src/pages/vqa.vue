<template>
    <div>
        <el-container>
            <NavigationBar/>
            <el-container class="parent-div">
                <VqaMenu/>
                <keep-alive>
                    <router-view v-if="$route.meta.keepAlive"/>
                </keep-alive>
                <router-view v-if="!$route.meta.keepAlive"/>
            </el-container>
        </el-container>
    </div>
</template>

<script>
    import NavigationBar from "../components/NavigationBar";
    import VqaMenu from "../components/VqaMenu";
    export default {
        name: "vqa",
        components: {
            NavigationBar,
            VqaMenu
        }
    }
</script>

<style scoped>
    .parent-div {
        position: fixed;
        top: 0;

        /*left: 30px;*/
        /*right: 30px;*/

        left: 0;
        right: 0;

        bottom: 0;
    }

    .el-main {
        padding-left: 260px;
        padding-top: 80px;
    }

    .el-scrollbar {
        height: 100%;
    }
</style>
<template>
    <el-aside width="240px" class="scrollable-div">
        <el-scrollbar ref="vqaMenuScrollBar">
            <el-menu
                    router
                    @select=""
                    @open=""
                    @close=""
                    :default-active="this.$route.path"
                    :default-openeds="['1','3']">
                <el-submenu index="1">
                    <template slot="title">Dataset Generation</template>
                    <el-menu-item-group>
                        <el-menu-item index="/vqa/dataset">Data Upload</el-menu-item>
                        <el-menu-item index="/vqa/label">Labeling Module</el-menu-item>
                    </el-menu-item-group>
                </el-submenu>
                <el-menu-item index="/vqa/modelEvaluation">Model Practice</el-menu-item>
                <el-submenu index="3">
                    <template slot="title">Result Representation</template>
                    <el-menu-item-group>
                        <el-menu-item index="/vqa/report">Evaluation Report</el-menu-item>
                        <el-menu-item index="/vqa/AI">AI Robot</el-menu-item>
                    </el-menu-item-group>
                </el-submenu>
            </el-menu>
        </el-scrollbar>
    </el-aside>
</template>

<script>
    export default {
        name: "VqaMenu",
        data() {
            return {
                category: []
            };
        },
        methods: {
        },
        created() {
        }
    }
</script>

<style scoped>
    .el-aside {
        position: fixed;
        top: 80px;
        bottom: 0;
        background-color: rgb(238, 241, 246);
        z-index: 999;
    }

    .el-scrollbar {
        height: 100%;
    }
</style>

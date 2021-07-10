<template>
  <el-main>
    <div class="top-bar">
      <div class="option-bar">
        <div class="header-bar">
          <span>My Dataset</span>
        </div>
        <div class="operation-btn-bar">
          <el-button type="primary" @click="newDatasetDialog"
            >Generate Dataset</el-button
          >
        </div>
      </div>
    </div>
    <el-dialog
      title="Generate Dataset"
      :visible.sync="addDatasetFormVisible"
      width="640px"
      append-to-body
      :close-on-click-modal="false"
      :close-on-press-escape="false"
      :show-close="false"
    >
      <el-steps :active="stepsActive" finish-status="success" align-center>
        <el-step title="Upload your data" icon="el-icon-upload" description="Create dataset"></el-step>
      </el-steps>
      <el-form
        ref="addDatasetForm"
        :model="addDatasetForm"
        :rules="rules"
        label-position="left"
        label-width="130px"
      >
        <el-form-item label="Name" prop="name">
          <el-input v-model="addDatasetForm.name"></el-input>
        </el-form-item>
        <el-form-item label="Description" prop="description">
          <el-input
            type="textarea"
            :rows="5"
            v-model="addDatasetForm.description"
          ></el-input>
        </el-form-item>
        <el-form-item label="Train/Valid/Test" prop="test">
          <template>
            <div class="block">
              <el-slider
                  v-model="addDatasetForm.ratio"
                  range
                  show-stops
                  :max="10">
              </el-slider>
            </div>
          </template>
        </el-form-item>
        <el-form-item>
          <el-upload class="upload-demo"
                     ref="upload"
                     drag
                     action="http://localhost:8089/upload"
                     multiple
                     :auto-upload="false"
                     :limit="5"
                     :on-success="handleFilUploadSuccess"
                     :on-remove="handleRemove"
          >
<!-- -->
<!--          <el-upload-->
<!--              class="upload-demo"-->
<!--              drag-->
<!--              ref="upload"-->
<!--              action="http://localhost:8089/upload"-->
<!--              :auto-upload="false"-->
<!--              :limit="5"-->
<!--              :on-success="handleFilUploadSuccess"-->
<!--              :on-remove="handleRemove"-->
<!--              multiple>-->
            <i class="el-icon-upload"></i>
            <div class="el-upload__text">drop files here,or <em>click here to upload files</em></div>
            <div class="el-upload__tip" slot="tip">please upload zip files</div>
          </el-upload>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button type="primary" @click="addDataset">Submit</el-button>
        <el-button @click="cancelAddDataset">Cancel</el-button>
      </div>
    </el-dialog>

    <el-dialog
      title="Upload Dataset"
      :visible.sync="uploadDatasetFormVisible"
      width="640px"
      append-to-body
      :close-on-click-modal="false"
      :close-on-press-escape="false"
      :show-close="false"
    >
      <el-steps :active="stepsActive" finish-status="success" align-center>
        <el-step title="Step 1" description="Create dataset"></el-step>
        <el-step title="Step 2" description="Go labeling"></el-step>
      </el-steps>

      <div slot="footer" class="dialog-footer">
        <!--        <el-button type="primary" @click="submitUpload">Submit</el-button>-->
        <el-button @click="cancelUploadDataset">Finish</el-button>
      </div>
    </el-dialog>

    <div class="content-div scrollable-div">
      <el-scrollbar>
        <div class="content-parent">
          <div class="content">
            <div class="item-list">
              <el-table
                ref="datasetTable"
                :data="datasetTableData"
                :max-height="datasetTableHeight"
              >
                <template slot="empty">
                  <span>{{ datasetTableEmptyText }}</span>
                </template>
                <el-table-column
                  prop="name"
                  label="Name"
                  min-width="20%"
                  align="center"
                  :show-overflow-tooltip="true"
                ></el-table-column>
                <el-table-column
                  prop="description"
                  label="Description"
                  min-width="30%"
                  align="center"
                  :show-overflow-tooltip="true"
                ></el-table-column>
                <el-table-column
                  prop="status"
                  label="Status"
                  min-width="15%"
                  align="center"
                >
                  <template slot-scope="scope">

                      <el-tag
                        slot="reference"
                        :type="scope.row.status ? 'success' : 'danger'"
                        @click="showMsg(scope.row.status)"
                        >{{
                          scope.row.status ? "Available" : "Not Available"
                        }}</el-tag
                      >
                  </template>
                </el-table-column>
                <el-table-column
                  prop="operations"
                  label="Operations"
                  min-width="35%"
                  align="center"
                >
                  <template slot-scope="scope">
                    <div v-if="scope.row.status">

                      <el-button
                        type="primary"
                        plain
                        size="small"
                        @click="goDownload(scope.row.link)"
                        >Download</el-button
                      >
                    </div>
                    <div v-else>

                      <el-button
                          type="primary"
                          plain
                          size="small"
                          @click="goDownload()"
                          disabled
                      >Download</el-button
                      >
<!--                      <el-button-->
<!--                        type="primary"-->
<!--                        plain-->
<!--                        size="small"-->
<!--                        @click="-->
<!--                          openUploadDatasetForm(-->
<!--                            scope.row.id,-->
<!--                            scope.row.trainSetName,-->
<!--                            scope.row.testSetName,-->
<!--                            scope.row.devSetName-->
<!--                          )-->
<!--                        "-->
<!--                        >Upload</el-button-->
<!--                      >-->
                    </div>
                  </template>
                </el-table-column>ratio
              </el-table>
            </div>
          </div>
        </div>
      </el-scrollbar>
    </div>
  </el-main>
</template>

<script>
export default {
  name: "dataset",
  data() {
    return {
      datasetTableHeight: 200,
      datasetTableData: [
        {},
      ],
      datasetTableEmptyText: "Loading...",
      stepsActive: 0,
      addDatasetFormVisible: false,
      addDatasetForm: {
        name: "",
        description: "",
        ratio:[4,8]
      },
      rules: {
        name: [
          { required: true, message: "Please input name", trigger: "blur" },
          // {min: 1, max: 10, message: 'Name length is between 1 and 10', trigger: 'blur'}
        ],
        description: [
          {
            required: true,
            message: "Please input description",
            trigger: "blur",
          },

        ],
        train: [
          {
            required: true,
            message: "Please input description",
            trigger: "blur",
          },

        ],
        valid: [
          {
            required: true,
            message: "Please input description",
            trigger: "blur",
          },

        ],
        test: [
          {
            required: true,
            message: "Please input description",
            trigger: "blur",
          },

        ],
      },
      uploadDatasetFormVisible: false,
      uploadDatasetForm: {
        datasetId: 0,
        type: "",
      },

      trainSetExist: false,
      testSetExist: false,
      devSetExist: false,
      trainSetFileList: [],
      testSetFileList: [],
      devSetFileList: [],
      fileBook: null
    };
  },
  methods: {
    handleRemove(file,fileList) {
      console.log(file,fileList);
    },
    submitUpload() {
      this.$refs.upload.submit();
    },
    // 文件上传成功时的函数
    handleFilUploadSuccess (res,file,fileList) {
      console.log(res,file,fileList)
      this.$message.success("上传成功")
    },
    handleUpdate () {
      this.dialogVisible = true;
    },
    // 处理文件上传的函数
    handleUpload () {
      // console.log(res,file)
      this.submitUpload()
      this.dialogVisible = false
    },

    download(url){
      const ele = document.createElement('a');
      ele.setAttribute('href', this.$options.filters['filterUrl'](url));
      //this.$options.filters['filterUrl']是调用全局过滤器,filterUrl是你自己项目main.js里面定义的过滤器
      ele.setAttribute('download',name);
      ele.style.display = 'none';
      document.body.appendChild(ele);
      ele.click();
      document.body.removeChild(ele);
    },


    goDownload(link){
      window.location.replace ("https://"+ link );
      // window.open(link_res[0])

      console.log(link)
    },
    loadDataset() {
      this.$axios({
        method: "get",
        url: "/dataSets",
      })
        .then((res) => {
          console.log(res.data);
          this.datasetTableData = res.data;
          if (this.datasetTableData.length === 0) {
            this.datasetTableEmptyText = "No Dataset";
          }
        })
        .catch((error) => {
          console.log(error);
        });
    },
    newDatasetDialog() {
      this.stepsActive = 0;
      this.addDatasetFormVisible = true;
    },
    addDataset() {
      this.handleUpload();
      let params = new FormData();

      params.append("name", this.addDatasetForm.name);
      params.append("description", this.addDatasetForm.description);
      params.append("train", this.addDatasetForm.ratio[0]);
      params.append("valid", this.addDatasetForm.ratio[1]-this.addDatasetForm.ratio[0]);
      params.append("test", 10-this.addDatasetForm.ratio[1]);
      // params.append("file", this.fileBook)
      this.$axios({
        method: "post",
        // url: "/" + this.$store.state.user.id + "/dataset/upload",
        url: "/addDataset",
        data:params,
        // data: {
        //   name: "xxx",//this.addDatasetForm.name,
        //   description: "xxx"//this.addDatasetForm.description,
        // },
      })
        .then((res) => {
          this.addDatasetFormVisible=true
          console.log(res.data);
          this.closeAddDatasetForm();
          this.loadDataset();
          this.$notify({
            title: "Success",
            message: "Successfully create dataset! ",
            type: "success",
          });
          this.openUploadDatasetForm(res.data, "", "", "");
        })
        .catch((error) => {
          console.log(error);
        });
    },
    cancelAddDataset() {
      this.closeAddDatasetForm();
    },
    closeAddDatasetForm() {
      this.addDatasetFormVisible = false;
      this.$refs["addDatasetForm"].resetFields();
    },
    submitUploadTrainSetFile() {
      // if (this.trainSetFileList.length === 0) {
      //     this.$notify.info({
      //         title: 'Notification',
      //         message: 'Please choose a file! '
      //     });
      // }
      // console.log(this.trainSetFileList);
      this.uploadDatasetForm.type = "train";
      this.$refs.uploadTrainSet.submit();
    },
    submitUploadTestSetFile() {
      this.uploadDatasetForm.type = "test";
      this.$refs.uploadTestSet.submit();
    },
    submitUploadDevSetFile() {
      this.uploadDatasetForm.type = "dev";
      DevSet.submit();
    },
    uploadDatasetFile(content) {
      let params = new FormData();
      params.append("dId", this.uploadDatasetForm.datasetId);
      params.append("type", this.uploadDatasetForm.type);
      // params.append("name", this.uploadDatasetForm.name);
      params.append("file", content.file);
      // console.log(content.file);

      this.$axios({
        method: "post",
        url: "/doctor/" + this.$store.state.user.id + "/dataSets",
        data: params,
      })
        .then((res) => {
          // console.log(res.data);
          this.$notify({
            title: "Success",
            message: "Successfully upload file! ",
            type: "success",
          });
        })
        .catch((error) => {
          console.log(error);
          this.$notify.error({
            title: "Error",
            message: "Wrong file format! ",
          });
        });
    },
    cancelUploadDataset() {
      this.$router.push('/vqa/label');

      this.closeUploadDatasetForm();
      this.loadDataset();
    },
    openUploadDatasetForm(datasetId, trainSet, testSet, devSet) {
      this.stepsActive = 1;
      this.uploadDatasetForm.datasetId = datasetId;
      this.trainSetExist = trainSet !== "";
      this.testSetExist = testSet !== "";
      this.devSetExist = devSet !== "";
      this.uploadDatasetFormVisible = true;
    },
    closeUploadDatasetForm() {
      this.uploadDatasetFormVisible = false;
      this.$refs["uploadDatasetForm"].resetFields();
      this.trainSetFileList = [];
      this.testSetFileList = [];
      this.devSetFileList = [];
    },
    showMsg(status) {
      if (status) {
        this.$notify({
          title: "Success",
          message: "Dataset is available! Begin a TASK or RESET dataset! ",
          type: "success",
          // offset: 200
        });
      } else {
        this.$notify.warning({
          title: "Warning",
          message: "Dataset is not available! Please UPLOAD first! ",
        });
      }
    },
    goToModelEvaluation(datasetId) {
      // this.$message({
      //     type: 'success',
      //     message: 'Dataset id: ' + datasetId + ', go to Model Evaluation page'
      // });
      this.$router.push("/deepLearning/modelEvaluation");
    },
    goToAutoSelection(datasetId) {
      // this.$message({
      //     type: 'success',
      //     message: 'Dataset id: ' + datasetId + ', go to Auto Selection page'
      // });
      this.$router.push("/deepLearning/autoSelection");
    },
  },
  created() {
    this.loadDataset();
  },
  mounted() {
    this.$nextTick(function () {
      this.datasetTableHeight =
        window.innerHeight - this.$refs.datasetTable.$el.offsetTop - 190;
      let self = this;
      window.onresize = function () {
        self.datasetTableHeight =
          window.innerHeight - self.$refs.datasetTable.$el.offsetTop - 190;
      };
    });
  },
};
</script>

<style scoped>
.el-main {
  position: fixed;
  top: 80px;
  left: 260px;
  right: 30px;
  bottom: 0;
  margin: 20px;
  padding: 10px;
}

.option-bar {
  display: inline-block;
  position: fixed;
  left: 280px;
  right: 40px;
}

.header-bar {
  float: left;
  margin: 0 20px;
}

.header-bar span {
  font-size: 20px;
  font-weight: bold;
  float: left;
}

.operation-btn-bar {
  float: right;
  margin: 0 20px;
  overflow: hidden;
}

.el-form {
  padding: 0 10px;
}

/*.el-upload {*/
/*  display: inline-block;*/
/*}*/

.content-div {
  position: fixed;
  top: 130px;
  left: 280px;
  right: 40px;
  bottom: 20px;
  z-index: -100;
}

.content {
  margin: 10px;
  padding: 30px;
}

.dataset-btn {
  margin: 0 auto;
}

.browse-btn {
  width: 70px;
  height: 30px;
  margin: 5px;
  vertical-align: top;
}

.el-tag {
  width: 95px;
  text-align: center;
}

.el-steps {
  margin-bottom: 40px;
}
</style>

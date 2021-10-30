<template>

  <el-main>
    <div class="top-bar">
      <div class="option-bar">
        <el-header class="header-bar">
          <div>
            <span>Evaluation Report</span>
            <el-row type="flex" justify="end">
                <el-button-group>
                  <el-button type="danger" size="mini" @click="deleteFailed()">Quickly Remove<i class="el-icon-delete el-icon--left"></i></el-button>
                  <el-button type="primary" size="mini" @click="downloadReports()">Download<i class="el-icon-download el-icon--right"></i></el-button>
                </el-button-group>

            </el-row>

          </div>
        </el-header>
        <div>
          <el-card class="box-card">
            <el-table
                :data="tableData1"
                style="width: 100%"
                :row-class-name="tableRowClassName"
                :default-sort="{prop: 'date', order: 'descending'}"

            >

              <el-table-column type="expand">
                <template slot-scope="props">
                  <el-form label-position="left" inline class="demo-table-expand">-->
                    <el-form-item label="Precision">
                      <span>{{props.row.prec}} </span>
                    </el-form-item>
                    <el-form-item label="Recall">
                      <span>{{props.row.recall}} </span>

                    </el-form-item>
                    <el-form-item label="F1-score">
                      <span>{{props.row.f1}} </span>

                    </el-form-item>

                    <el-form-item label="BLEU">
                      <span>{{props.row.bleu}}</span>

                    </el-form-item>
                  </el-form>
                </template>

              </el-table-column>

              <el-table-column
                  prop="name"
                  label="Report"
                  width="180"
                  sortable
              ></el-table-column>

              <el-table-column
                  prop="classification"
                  label="Model"
                  width="180"
                  sortable
              ></el-table-column>

              <el-table-column
                  prop="date"
                  label="Generated Date"
                  width="180"
                  sortable
              >
              </el-table-column>
              <el-table-column prop="state" label="Status" width="180" sortable>
              </el-table-column>
              <el-table-column
                  prop="data"
                  label="DataSet"
                  width="180"
                  sortable
              >
              </el-table-column>

              <el-table-column
                  prop="epoch"
                  label="Epochs"
                  width="180"
              >
              </el-table-column>

              <el-table-column
                  prop="batch"
                  label="Batchsize"
                  width="180"
              >
              </el-table-column>
            </el-table>


          </el-card>
        </div>
      </div>
      <!-- </div> -->
    </div>

  </el-main>
</template>

<script>
export default {
  mounted() {
    this.getAllReports()
  },
  methods: {
    fileChange(e) {
      try {
        const fu = document.getElementById('file')
        if (fu == null) return
        this.form.imgSavePath = fu.files[0].path
        console.log(fu)
      } catch (error) {
        console.debug('choice file err:', error)
      }
    },
    btnChange() {
      var file = document.getElementById('file')
      file.click()
    },
    downloadReports(){

      // this.stepsActive = 0;
      // this.downloadVisable = true;

      this.$axios({
        method: 'get',
        url: '/downloadReports',
      }).then(res => {
        console.log(res.data);
        const content=res.data;
        const blob=new Blob([content]);
        const fileName='report.csv';
        if('download' in document.createElement('a')){
          const link=document.createElement('a')
          link.download=fileName
          link.style.display='none'
          link.href=URL.createObjectURL(blob)
          document.body.appendChild(link)
          link.click()
          URL.revokeObjectURL(link.href)
          document.body.removeChild(link)
        }else{
          navigator.msSaveBlob(blob,fileName)
        }
      }).catch(error => {
        console.log(error);
        alert("ERROR! Load Reports Failed! ");
      });
    },
    deleteFailed(){

      // this.stepsActive = 0;
      // this.downloadVisable = true;

      this.$axios({
        method: 'delete',
        url: '/deleteFailed',
      }).then(res => {
        console.log(res.data);
        alert("Delete failed models sucess! Please refresh manually.");
        // const content=res.data;
        // const blob=new Blob([content]);
        // const fileName='report.csv';
        // if('download' in document.createElement('a')){
        //   const link=document.createElement('a')
        //   link.download=fileName
        //   link.style.display='none'
        //   link.href=URL.createObjectURL(blob)
        //   document.body.appendChild(link)
        //   link.click()
        //   URL.revokeObjectURL(link.href)
        //   document.body.removeChild(link)
        // }else{
        //   navigator.msSaveBlob(blob,fileName)
        // }
      }).catch(error => {
        console.log(error);
        alert("ERROR! Delete Reports Failed! ");
      });
    },
    tableRowClassName({row, rowIndex}) {
      if (row.state=="done"){
        return "success-row";
      }
      else if (row.state=="running"){
        return "warning-row";
      }
      else {
        return 'failed-row';
      }
      // if (rowIndex === 1) {
      //   return "warning-row";
      // } else if (rowIndex === 3) {
      //   return "success-row";
      // }
      return "";
    },
    getAllReports(){
      this.$axios({
        method: 'get',
        url: '/reports',
      }).then(res => {
        console.log(res.data);
        this.tableData1=res.data;
        for(var i=0;i<this.tableData1.length;i++){
          if(this.tableData1[i].classification=="NLM"||this.tableData1[i].classification=="VGG-Seq2Seq"){
            this.tableData1[i].prec="---";
            this.tableData1[i].recall="---";
            this.tableData1[i].f1="---";
            this.tableData1[i].bleu=this.tableData1[i].bleu;
            continue;
          }else {
            this.tableData1[i].prec=this.tableData1[i].prec/100+"%";
            this.tableData1[i].recall=this.tableData1[i].recall/100+"%";
            this.tableData1[i].f1=this.tableData1[i].f1/100+"%";
            this.tableData1[i].bleu="---";
          }

          // this.tableData1[i].bleu=this.tableData1
        }
        // this.tableData1.row.prec=this.tableData1.row.prec/100;

      }).catch(error => {
        console.log(error);
        alert("ERROR! Load Reports Failed! ");
      });
    },
  },
  data() {
    return {
      downloadVisable:false,
      form: {
        name: "",
        desc: "",
        imgSavePath:""
      },
      characterList: [
        {
          options: [
            {value: 1, label: "Joint Embedding approches"},
            {value: 2, label: "Attention mechanisms"},
            {value: 3, label: "Compositional Models"},
            {value: 4, label: "Models using external knowledge base"},
          ],
        },
      ],
      tableData1: [
      ],
      models:[
        {name:'NLM',
          type:"bleu",
        },
//         {name:'CR',
//           type:"acc",
// },
        {name:'VGG-Seq2Seq',
          type:"bleu",

        },
        {name:'ODL',
          type:"acc",
        },
//         {name:'CGMVQA',
//           type:"acc",
// },
        {name:'MMBERT',
          type:"acc",
        },
        {name:'Knowledge Embedded Metalearning',
          type:"acc",
        },
      ]
    };
  },
};
</script>

<style>
.el-table .warning-row {
  background: oldlace;
}

.el-table .success-row {
  background: #f0f9eb;
}

.el-table .failed-row{
  background: rgba(255, 101, 0, 0.38);
}

.header-bar span {
  font-size: 20px;
  font-weight: bold;
  float: left;
}

.el-header {
  background-color: #ffffff;
  color: #333;
  line-height: 60px;
}

.demo-table-expand {
  font-size: 0;
}

.demo-table-expand label {
  width: 90px;
  color: #99a9bf;
}

.demo-table-expand .el-form-item {
  margin-right: 0;
  margin-bottom: 0;
  width: 20%;
}


</style>


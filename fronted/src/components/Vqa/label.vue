<template>
  <el-main>
    <div class="top-bar">
      <div class="option-bar">
        <div class="header-bar">
          <span>Labeling Module</span>
        </div>
        <el-row type="flex" justify="end">
          <el-select v-model="characterChosen" placeholder="Please choose dataset."
                     @change="chooseCharacter">
            <el-option
                v-for="item in characterList_dataset"
                :key="item.value"
                :label="item.label"
                :value="item.value">
            </el-option>
          </el-select>
        </el-row>

      </div>
    </div>
    <div>

    </div>
    <template>

      <el-row type="flex" justify="end">
        <el-button-group>
          <!--                  <el-button type="primary" size="mini" icon="el-icon-edit">Edit</el-button>-->
          <el-button type="primary" size="medium" @click="generate()">Submit<i class="el-icon-check"></i></el-button>
        </el-button-group>
      </el-row>
    </template>
  <div>
    <el-table :data="patientIds.slice((currentPage-1)*pageSize,currentPage*pageSize)" style="width: 100%">
      <el-table-column prop="name" label="Document ID">
        <template slot-scope="scope">
          <span>{{scope.row.patientId}}</span>

        </template>
      </el-table-column>
      <el-table-column prop="name" label="Picture ID">
        <template slot-scope="scope">
          <span>{{scope.row.photoId}}</span>
        </template>
      </el-table-column>

      <el-table-column label="Operation">
        <template slot-scope="scope">
          <el-popover
              placement="bottom"
              title="Check if the image corresponds to the disease (do not check otherwise)."
              width="1000"
              trigger="click"
              content="">
            <el-button slot="reference" @click="getPhoto(scope.$index)">Edit</el-button>

            <div>
              <el-scrollbar style="height:500px">
                <!--                                <p style="font-size:20px">请勾选每种诊断疾病对应的检查图像(不相关的请勿勾选)：</p>-->
                <p style="font-size:15px">Symptoms：</p>
                <p style="font-size:15px">{{scope.row.sym}}</p>
                <p style="font-size:15px">Auxiliary examination：</p>
                <p style="font-size:15px">{{scope.row.annotation}}</p>
                <!--                                <p style="font-size:15px">疾病：</p>-->
                <!--                                <p style="font-size:15px">{{scope.row.dia}}</p>-->
                <!--                                <button @click="getPhoto(scope.$index)">加载图片</button>-->
                <el-input placeholder="Please enter the description of the inspection image (refer to the auxiliary inspection)." v-model="description"  style="width:90%; float:left">
                </el-input>
                <div></div>
                <div class="option-block">
                  <el-select v-model="boneChosen" multiple filterable allow-create default-first-option placeholder="Please select the location" >
                    <el-option-group v-for="group in boneList" :key="group.label" :label="group.label">
                      <el-option v-for="item in group.options" :key="item.value" :label="item.label" :value="item.value">
                      </el-option>
                    </el-option-group>
                  </el-select>
                </div>
                <div class="option-block">
                  <el-select v-model="typeChosen" multiple filterable allow-create default-first-option placeholder="Please select the type of examination sheet" >
                    <el-option-group v-for="group in typeList" :key="group.label" :label="group.label">
                      <el-option v-for="item in group.options" :key="item.value" :label="item.label" :value="item.value">
                      </el-option>
                    </el-option-group>
                  </el-select>
                </div>
                <div class="option-block">
                  <el-select v-model="xrayChosen" multiple filterable allow-create default-first-option placeholder="Please select location identification" >
                    <el-option-group v-for="group in xrayList" :key="group.label" :label="group.label">
                      <el-option v-for="item in group.options" :key="item.value" :label="item.label" :value="item.value">
                      </el-option>
                    </el-option-group>
                  </el-select>
                </div>
                <div class="option-block">
                  <el-select v-model="DirectionChosen" multiple filterable allow-create default-first-option placeholder="Please scroll down to select" >
                    <el-option-group v-for="group in characterList" :key="group.label" :label="group.label">
                      <el-option v-for="item in group.options" :key="item.value" :label="item.label" :value="item.value">
                      </el-option>
                    </el-option-group>
                  </el-select>
                </div>
                <label>
                  <img :src="imgPath" width="500px" height="400px"/>
                </label>
                <div class="checkbox" v-for="(k,n) in dias">
                  <label >
                    <!--                                        <label class="checkbox-inline col-md-4" style="margin-left:0">-->
                    <input type="checkbox" :value="k" name="ct"  style="width:20px;height:20px;">
                    <p>{{k}}</p>
                  </label>
                </div>


                <p style="text-align:center;">
                  <button  @click="getId(scope.row.patientId,scope.row.photoId,description,boneChosen,DirectionChosen,typeChosen,xrayChosen)">Submit</button>

                </p>
              </el-scrollbar>
            </div>
          </el-popover>

        </template>
      </el-table-column>
    </el-table>
    <!--分页区域-->
    <!--        <div class="block" style="margin-top:15px;">-->
    <!--            <el-pagination align='center' @size-change="handleSizeChange" @current-change="handleCurrentChange" :current-page="currentPage" :page-sizes="[1,5,10,20]" :page-size="pageSize" layout="total, sizes, prev, pager, next, jumper" :total="patientIds.length">-->
    <!--            </el-pagination>-->
    <!--        </div>-->
    <div class="yema">
          <el-pagination background
                         @size-change="handleSizeChange"
                         @current-change="handleCurrentChange"
                         :current-page="currentPage"
                         :page-sizes="[5,10,15]"
                         :page-size="pagesize"
                         layout="total,jumper,prev, pager, next,sizes"
                         :total="patientIds.length" >
          </el-pagination>
    </div>
  </div>
    </el-main>
</template>

<script>
export default {
  data() {
    return {
      characterList_dataset: [],
      imgPath:'',
      characterChosen:"",
      upList:[],
      description:'',
      boneName:'',
      photoFile:'',
      dias:[],
      patientIds:[],
      pngFile:'1.png',
      currentPage: 1, // 当前页码
      total: 20, // 总条数
      pageSize: 100 ,// 每页的数据条数
      DirectionChosen:'Please select the shooting surface',
      characterList: [{
        options: [
          {value:1, label: "Transverse"},//横断面
          {value:2, label: "Coronal"},//冠状面
          {value:3, label: "Sagittal"}//矢状面
        ]
      }],
      typeChosen:'Please select the type of image to check',
      typeList: [{
        options: [
          {value:1, label: "ct"},
          {value:2, label: "x-ray"}
        ]
      }],
      boneChosen:'Please select the organ corresponding to the bone',
      boneList: [{
        options: [
          {value:1, label: "Head"},
          {value:2, label: "Chest"},
          {value:3, label: "Hand"},
          {value:4, label: "Leg"},
        ]
      }],
      xrayChosen:'Please choose the direction of shooting',
      xrayList: [{
        options: [
          {value:1, label: "A-P"},//（前后位）
          {value:2, label: "Lateral"},//（侧位）
          {value:3, label: "Lordotic"},//（斜位）
          {value:4, label: "A-p supine"},//（前后位 仰卧）
          {value:5, label: "P-A "},//（后前位）
        ]
      }]
    }
  },
  methods: {
    opens() {
      this.$alert('Labeling is done, Now it\'s generation VQA pairs, pleases wait for a moment.', 'Information', {
        confirmButtonText: 'Confirm',
        // callback: action => {
        //   this.$message({
        //     type: 'info',
        //     message: `action: ${ action }`
        //   });
        // }
      });
    },
    generate(){
      this.$axios({
        method: 'get',
        url: '/doneLableing/'+this.characterList_dataset[this.characterChosen-1].label,

      }).then(res => {
        this.opens()
      }).catch(error => {
        this.opens()
        this.$router.push("/vqa/dataset")
      });
    },
    chooseCharacter(value) {
      // console.log(value)
      // console.log("dataset name: ")
      // console.log(this.characterList_dataset[0].label)
      this.characterChosen=value;
      this.getAllPatients()
      // getAllModels()
    },
    getAllModels(){
      this.$axios({
        method: 'get',
        url: '/dataSetsNames',
      }).then(res => {
        console.log("all datasets: ")
        console.log(res.data)
        for(var i=0;i<res.data.length;i++){
          // console.log(res.data[i])
          var temp={"value":i+1,"label":res.data[i]}

          // console.log(temp)
          this.characterList_dataset.push(temp)
        }
        // this.characterChosen=this.characterList[0].label;

      }).catch(error => {
        console.log(error);
        alert("ERROR! Load Models Failed! ");
      });
    },
    handleSizeChange(val) {
      console.log(`每页 ${val} 条`);
      // this.currentPage = 1;
      this.pageSize = val;
    },
    handleCurrentChange(val) {
      console.log(`当前页: ${val}`);
      this.currentPage = val;
    },
    getId(p1,p2,description,boneName,DirectionChosen,typeChosen,xrayChosen){

      console.log("patientId",p1);
      console.log("photoId",p2);
      var day = document.getElementsByName('ct');
      // var week=document.getElementById('week').getAttribute("value");
      var checkArr = [];

      for (var k in day) {
        if (day[k].checked){
          checkArr.push(day[k].value);

        }
      }
      console.log("diaList",checkArr);
      console.log("description", description);
      console.log("boneName", boneName);
      console.log("direction", DirectionChosen);
      console.log("type", typeChosen);
      console.log("position", xrayChosen);
      // console.log('checkArr', checkArr);
      // var patientId=document.getElementById("patient").getAttribute("value");
      // var dia=document.getElementById("disease").getAttribute("value");

      // console.log('numList', checkArr);
      // console.log('patientId', patientId);
      // console.log('disease', dia);
      let params = new FormData();
      params.append("patientId", p1);
      params.append("diaList", checkArr);
      params.append("photoId", p2);
      params.append("description", description);
      params.append("boneName", boneName);
      params.append("direction", DirectionChosen);
      params.append("type", typeChosen);
      params.append("position", xrayChosen);


      //数据库连接对象
      this.$axios({
        method: 'post',
        url: '/vqa/'+this.characterList_dataset[this.characterChosen-1].label,
        data: params
      }).then(res => {
        this.$message({
          message: 'Successfully!',
          type: 'success',
          offset:60,
          showClose: true
        });
        location.reload();

        // this.message="提交完成";



      }).catch(error => {
        console.log(error);
      });
    },
    getPhoto(id){
      console.log("id",id);
      this.photoFile=this.patientIds[id].photoId + '.jpg';
      this.imgPath = './../../../static/'+this.characterList_dataset[this.characterChosen-1].label+ '/' + this.photoFile ;

      console.log("imgpath",this.imgPath);

      this.dias=this.patientIds[id].dias[0].split(",");
      console.log("dias",this.dias);
      console.log("diasType",typeof (this.dias[0]));

    },
    confirmEdit(index,row){
      row.edit = false;
      this.$message({
        message: '该地址已经成功修改',
        type: 'success'
      })
    },getAllPatients(){
      this.$axios({
        method: 'get',
        url: '/all/'+this.characterList_dataset[this.characterChosen-1].label,
      }).then(res => {
        console.log("dataset name: "+this.characterList_dataset[this.characterChosen-1].label)
        this.patientIds=[]
        for (let row in res.data) {
          if (res.data.hasOwnProperty(row)) {
            var text = res.data[row].diaList;
            var array = text.split("|");

            this.patientIds.push({
              patientId: res.data[row].patientId,
              sym:res.data[row].sym ,
              photoId: res.data[row].photoId,
              dias:array,
              annotation:res.data[row].annotation
            });
          }
        }

      }).catch(error => {
        console.log(error);
      });
    }
  },mounted(){
    this.getAllModels()
    // this.getAllPatients();

  }
}
</script>
<style scoped>
.d-radio {
  padding-left: 0;
  padding-bottom: 10px;
  margin-bottom: 5px;
  border-bottom: 1px solid #e2e2e2;
}

.d-radio .col-md-12 {
  padding-left: 0;
}
.el-scrollbar__wrap{
  overflow-x: hidden;
}
.el-message {
  top:350px !important;
  z-index: 99999 !important;
}

</style>
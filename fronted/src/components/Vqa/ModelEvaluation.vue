<template>
  <el-main>
    <div class="top-bar">
      <div class="option-bar">
        <el-header class="header-bar">
          <div>
            <span>Model Practice</span>
          </div>
        </el-header>
        <div>
          <el-card class="box-card">
            <el-collapse v-model="activeNames" @change="handleChange">
              <div v-for="(item,index) in models">
                <el-collapse-item :title="item.name" :name="index">
                  <el-header>-->
                                      <!-- <div>Model Description</div> -->
                                      <el-link
                                        :href="item.link"
                                        type="primary"
                                        class="ms"
                                        >{{item.paper}}</el-link
                                      >
                                      <div>
                                      </div>
                                    </el-header>
                                    <div>
                                      <el-footer>
<!--                                        <div>-->
<!--                                          <span>&ndash;&gt;{{item.type}}</span>-->
<!--                                        </div>-->
                                        <div>
                                          <!-- <el-main> -->
                                          <span>Dataset</span>
                                          <div>
                                            <template>
                                              <el-select v-model="characterChosen" placeholder="Please"
                                                         @change="chooseDataset">
                                                <el-option
                                                    v-for="item in options"
                                                    :key="item.value"
                                                    :label="item.label"
                                                    :value="item.value">
                                                </el-option>
                                              </el-select>
                                            </template>
                                          </div>
                                          <span>Epoch</span>
                                          <div>
                                            <template>
                                              <div class="block">
                                                <el-slider
                                                    v-model="epoch"
                                                    :step="10"
                                                    show-stops
                                                    show-input>
                                                </el-slider>
                                              </div>
                                            </template>
                                          </div>
                                          <span>Batchsize</span>
                                          <div>
                                            <template>
                                              <el-select v-model="choosenBatchsize" placeholder="Please"
                                                         @change="chooseBatchsize"
                                                          >
                                                <el-option
                                                    v-for="item in batchsize"
                                                    :key="item.value"
                                                    :label="item.label"
                                                    :value="item.value">
                                                </el-option>
                                              </el-select>
                                            </template>
                                          </div>

                                          <span  v-if="item.rnn_cell_vis">Rnn Cell</span>
                                          <div>
                                            <template>
                                              <el-select v-model="choosenRnncell" placeholder="Please"
                                                         @change="chooseRnncell"
                                                         v-if="item.rnn_cell_vis">
                                                <el-option
                                                    v-for="item in rnn"
                                                    :key="item.value"
                                                    :label="item.label"
                                                    :value="item.value">
                                                </el-option>
                                              </el-select>
                                            </template>
                                          </div>
                                          <span v-if="item.embedding_vis">Embedding</span>
                                          <div>
                                            <template>
                                              <el-select v-model="choosenEmbedding" placeholder="Please"
                                                         @change="chooseEmbedding"
                                                         v-if="item.embedding_vis">
                                                <el-option
                                                    v-for="item in embedding"
                                                    :key="item.value"
                                                    :label="item.label"
                                                    :value="item.value">
                                                </el-option>
                                              </el-select>
                                            </template>
                                          </div>
                                          <span>TrainName</span>
                                          <div>
                                            <el-col :span="12">
                                              <el-input placeholder="Please input model name"
                                                        v-model="input" size="medium">
                                              </el-input>
                                            </el-col>

                                          </div>
                                          <div class="operation-btn-bar">
                                            <el-button type="primary" @click="trainModel(item)"
                                              >Train</el-button
                                            >
                                          </div>
                                          <div></div>
                                          <!-- </el-main> -->
                                        </div>
                                      </el-footer>
                                    </div>
                </el-collapse-item>
              </div>
            </el-collapse>
          </el-card>
        </div>
      </div>

      <!-- </div> -->
    </div>
  </el-main>
</template>

<script>
export default {
  methods: {
    getAllDatasets(){
      // options: [
      //   { value: 1, label: "VQA-Med-2019" },
      //   { value: 2, label: "VQA-RAD" },
      //   { value: 3, label: "MVQA(Our)" },
      // ],
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
          this.options.push(temp)
        }
        // this.characterChosen=this.characterList[0].label;

      }).catch(error => {
        console.log(error);
        alert("ERROR! Load Models Failed! ");
      });
    },
    chooseEmbedding(value){
      this.choosenEmbedding=value;
    },
    chooseRnncell(value){
      this.choosenRnncell=value;
    },
    chooseBatchsize(value){
      this.choosenBatchsize=value;
    },
    chooseDataset(value) {
      // console.log(label);
      this.characterChosen=value;
    },
    tableRowClassName({ row, rowIndex }) {
      if (rowIndex === 1) {
        return "warning-row";
      } else if (rowIndex === 3) {
        return "success-row";
      }
      return "";
    },
    trainModel(item){
      console.log(item)
      alert("Success!Begin training "+item.name+" model.");
      var dataset='';
      var batchsize="";
      var rnn="";
      var embedding="";
      //getDataset
      for(var i=0;i<this.options.length;i++){
        if(this.characterChosen==this.options[i].value){
          // console.log(this.options[i])
          dataset=this.options[i].label;
          break;
        }
      }
      //getBatchsize
      for(var i=0;i<this.batchsize.length;i++){
        if(this.choosenBatchsize==this.batchsize[i].value){
          // console.log(this.options[i])
          batchsize=this.batchsize[i].label;
          break;
        }
      }
      //rnn
      for(var i=0;i<this.rnn.length;i++){
        if(this.choosenRnncell==this.rnn[i].value){
          // console.log(this.options[i])
          rnn=this.rnn[i].label;
          break;
        }
      }
      //embedding
      for(var i=0;i<this.embedding.length;i++){
        if(this.choosenEmbedding==this.embedding[i].value){
          // console.log(this.options[i])
          embedding=this.embedding[i].label;
          break;
        }
      }
      console.log(dataset)
      this.$axios({
        method: 'post',
        url: '/train/'+item.name,
        data: {
          name: this.input,
          data: dataset,
          batchsize:batchsize,
          epoch:this.epoch,
          rnn_cell:rnn,
          embedding:embedding
        }
      }
      ).then(res => {
        console.log(this.characterList)
        console.log(res.data);
      }).catch(error => {
        console.log(error);
        alert("ERROR! Check Console plz! ");
        this.$axios({
              method: 'post',
              url: '/error/'+item.name,
            }
        ).then(res => {
          console.log(this.characterList)
          console.log(res.data);
        }).catch(error => {
          console.log(error);});
      });
    },
  },
  mounted() {
    this.getAllDatasets()

  },
  data() {

    return {
      choosenBatchsize:"",
      characterChosen:"",
      epoch:"",
      choosenRnncell:'',
      choosenEmbedding:'',
      input:'',
      options: [
        // { value: 1, label: "VQA-Med-2019" },
        // { value: 2, label: "VQA-RAD" },
        // { value: 3, label: "MVQA(Our)" },
      ],
      rnn:[
        { value: 1, label: "RNN" },
        { value: 2, label: "GRU" },
        { value: 3, label: "LSTM" },
      ],
      embedding:[
        { value: 1, label: "w2v" },
        { value: 2, label: "glove" },
        { value: 3, label: "none" },
      ],
      batchsize:[
        { value: 1, label: "1" },
        { value: 2, label: "4" },
        { value: 3, label: "8" },
        { value: 4, label: "16" },
        { value: 5, label: "32" },
        { value: 6, label: "64" },
        { value: 7, label: "128" },
      ],
      models:[
        {name:'NLM',
          paper:'NLM at VQA-Med 2020: Visual Question Answering and Generation in the Medical Domain',
          link:'http://ceur-ws.org/Vol-2696/paper_98.pdf',
          type:"Seq2Seq",
          embedding_vis:true,
          rnn_cell_vis:true
},
//         {name:'CR',
//           paper:'Medical Visual Question Answering via Conditional Reasoning',
//           link:'http://www4.comp.polyu.edu.hk/~csxmwu/papers/MM-2020-Med-VQA.pdf',
//           type:"attention",
//           embedding_vis:false,
//           rnn_cell_vis:false
// },
        {name:'VGG-Seq2Seq',
          paper:'JUST at VQA-Med: A VGG-Seq2Seq Model',
          link:'http://ceur-ws.org/Vol-2125/paper_171.pdf',
          type:"Seq2Seq",
          embedding_vis:false,
          rnn_cell_vis:false

},
        {name:'ODL',
          paper:'Overcoming Data Limitation in Medical Visual Question Answering. ',
          type:"joint embedding",
          link:'https://arxiv.org/abs/1909.11867',
          embedding_vis:false,
          rnn_cell_vis:false
},
//         {name:'CGMVQA',
//           paper:'CGMVQA: A New Classification and Generative Model for Medical Visual Question Answering',
//           link:'https://ieeexplore.ieee.org/abstract/document/9032109',
//           type:"attention",
//           embedding_vis:false,
//           rnn_cell_vis:false
// },
        {name:'MMBERT',
          paper:'MMBERT: Multimodal BERT Pretraining for Improved Medical VQA',
          link:'https://arxiv.org/abs/2104.01394',
          type:"attention",
          embedding_vis:false,
          rnn_cell_vis:false
},
        {name:'Knowledge Embedded Metalearning',
          paper:'Learning from the Guidance: Knowledge Embedded Meta-learning for Medical Visual Question Answering',
          link:'https://link.springer.com/chapter/10.1007/978-3-030-63820-7_22',
          type:"knowledge embedding",
          embedding_vis:false,
          rnn_cell_vis:false
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

.el-header {
  background-color: #ffffff;
  color: #333;
  line-height: 60px;
}

.ms {
  text-align: right;
}
</style>


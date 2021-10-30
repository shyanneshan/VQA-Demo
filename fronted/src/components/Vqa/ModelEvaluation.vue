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
            <el-collapse>
              <div v-for="(modeltype,typeindex) in modelType">
                <el-collapse-item :title="modeltype.name" :name="typeindex">
                  <el-collapse v-model="activeNames" @change="handleChange">
                    <div v-for="(item,index) in modeltype.model">
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
                              <span v-if="item.attention_vis">Attention</span>
                              <div>
                                <template>
                                  <el-select v-model="choosenAttention" placeholder="Please"
                                             @change="chooseAttention"
                                             v-if="item.attention_vis">
                                    <el-option
                                        v-for="item in attention"
                                        :key="item.value"
                                        :label="item.label"
                                        :value="item.value">
                                    </el-option>
                                  </el-select>
                                </template>
                              </div>
                              <span v-if="item.construct_vis">Construct Module</span>
                              <div>
                                <template>
                                  <el-select v-model="choosenConstruct" placeholder="Please"
                                             @change="chooseConstruct"
                                             v-if="item.construct_vis">
                                    <el-option
                                        v-for="item in construct"
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
        url: '/dataSetsNamesLabeled',
      }).then(res => {
        console.log("all datasets: ")
        console.log(res.data)
        for(var i=0;i<res.data.length;i++){
          // console.log(res.data[i])
          var temp={"value":i+1,"label":res.data[i]}

          // console.log(temp)
          this.options.push(temp)
        }
        console.log(this.options)
        // this.characterChosen=this.characterList[0].label;

      }).catch(error => {
        console.log(error);
        alert("ERROR! Load Models Failed! ");
      });
    },
    chooseEmbedding(value){
      this.choosenEmbedding=value;
    },
    chooseAttention(value){
      this.choosenAttention=value;
    },
    chooseConstruct(value){
      this.choosenConstruct=value;
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
      alert("Begin training "+item.name+" model.");
      var dataset='';
      var batchsize="";
      var rnn="";
      var embedding="";
      var attention="";
      var construct="";
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
      //attention
      for(var i=0;i<this.attention.length;i++){
        if(this.choosenAttention==this.attention[i].value){
          // console.log(this.options[i])
          attention=this.attention[i].label;
          break;
        }
      }
      //constructor
      for(var i=0;i<this.construct.length;i++){
        if(this.choosenConstruct==this.construct[i].value){
          // console.log(this.options[i])
          construct=this.construct[i].label;
          break;
        }
      }
      console.log(dataset)
      console.log(attention)
      console.log(construct)
      this.$axios({
        method: 'post',
        url: '/train/'+item.name,
        data: {
          name: this.input,
          data: dataset,
          batchsize:batchsize,
          epoch:this.epoch,
          rnn_cell:rnn,
          embedding:embedding,
          attention:attention,
          constructor:construct
        }
      }
      ).then(res => {
        if (res.data=="name_existed"){
          alert("ERROR! This name has existed! ");
          console.log("name_existed checking.");
          console.log(res.data);
        }
        // else {
        //   alert("Success!Begin training "+item.name+" model.");
        // }
        console.log(this.characterList)
        console.log(res.data);
      }).catch(error => {
        console.log(error);
        alert("TRAIN ERROR! Check Console plz! ");
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
      choosenBatchsize:32,
      characterChosen:1,
      epoch:20,
      choosenRnncell:1,
      choosenEmbedding:3,
      choosenAttention:1,
      choosenConstruct:3,
      input:'',
      options: [
        // { value: 1, label: "VQA-Med-2019" },
        // { value: 2, label: "VQA-RAD" },
        // { value: 3, label: "MVQA(Our)" },
      ],
      rnn:[
        // { value: 1, label: "RNN" },
        { value: 1, label: "GRU" },
        { value: 2, label: "LSTM" },
      ],
      embedding:[
        { value: 1, label: "w2v" },
        { value: 2, label: "glove" },
        { value: 3, label: "none" },
      ],
      attention:[
        { value: 1, label: "SAN" },
        { value: 2, label: "BAN" }
      ],
      construct:[
        { value: 1, label: "maml" },
        { value: 2, label: "autoencoder" },
        { value: 3, label: "both" },
        // { value: 4, label: "none" },
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
      modelType:[

        {name:'Joint Embedding  Models',
          model:[
            {name:'--->  ODL',
              paper:'Overcoming Data Limitation in Medical Visual Question Answering. ',
              type:"joint embedding",
              link:'https://arxiv.org/abs/1909.11867',
              embedding_vis:false,
              rnn_cell_vis:true,
              attention_vis:true,
              construct_vis:true
            },
          ]},
        {name:'Encoder-Decoder Models',
          model:[
            {name:'--->  NLM',
              paper:'NLM at VQA-Med 2020: Visual Question Answering and Generation in the Medical Domain',
              link:'http://ceur-ws.org/Vol-2696/paper_98.pdf',
              type:"Seq2Seq",
              embedding_vis:true,
              rnn_cell_vis:true,
              attention_vis:false,
              construct_vis:false
            },
            {name:'--->  VGG-Seq2Seq',
              paper:'JUST at VQA-Med: A VGG-Seq2Seq Model',
              link:'http://ceur-ws.org/Vol-2125/paper_171.pdf',
              type:"Seq2Seq",
              embedding_vis:false,
              rnn_cell_vis:false,
              attention_vis:false,
              construct_vis:false
            },
          ]},
        {name:'Attention-based  Models',
          model:[
            {name:'--->  MMBERT',
              paper:'MMBERT: Multimodal BERT Pretraining for Improved Medical VQA',
              link:'https://arxiv.org/abs/2104.01394',
              type:"attention",
              embedding_vis:false,
              rnn_cell_vis:false,
              attention_vis:false,
              construct_vis:false
            },
          ]},
        {name:'Knowledge Embedding  Models',
          model:[
            {name:'--->  ArticleNet',
              paper:'OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge',
              link:'https://openaccess.thecvf.com/content_CVPR_2019/papers/Marino_OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_External_Knowledge_CVPR_2019_paper.pdf',
              type:"knowledge embedding",
              embedding_vis:false,
              rnn_cell_vis:false,
              attention_vis:false,
              construct_vis:false
            },
          ]},
      ],
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


package com.mvqa.demo.entity;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.*;

@Component
public class PythonCallEntity {

    @Value("${model.torch}")
    private String torch_env;// = "/data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python";//""python3.6";

    @Value("${model.tf}")
    private String tf_env;// = "/data2/entity/bhy/VQADEMO/model_code/vgg_seq2seq/venv/bin/python";//""/home/wxl/anaconda3/envs/tf/bin/python";

    @Value("${model.dataset}")
    private String datasetdir;// = "/data2/entity/bhy/VQADEMO/dataset/";//""/home/wxl/Documents/VQADEMO/dataset/";

    @Value("${model.weight}")
    private String weightdir;// = "/data2/entity/bhy/VQADEMO/weights/";

    //MMBERT
    public String MMBERT_train(String data, String name, String epoch, String batch) throws IOException {

//        /data2/entity/bhy/VQADEMO/model_code/MMBERT/venv/bin/python \
//        -u /data2/entity/bhy/VQADEMO/model_code/MMBERT/train.py \
//        --run_name 'dataset' \
//        --data_dir 'dataset' \
//        --save_dir 'dataset' \
//        --batch_size 16 \
//        --epochs 10
        String dataset = " --data_dir " + datasetdir + data ;
        String model = " /data2/entity/bhy/VQADEMO/model_code/MMBERT/train.py ";
//            String m=" --mode train";
        String epoch_ = " --epochs " + epoch;
        String batch_ = " --batch_size " + batch;
        String n = " --run_name " + name;
        String save_dir = " --save_dir " + weightdir+name;
        String traincmd = torch_env + model + n + dataset + epoch_ + batch_+save_dir;
        System.out.println(traincmd);
        Runtime run = Runtime.getRuntime();
        try {
            Process process = run.exec(traincmd);
            InputStream in = process.getInputStream();
            InputStreamReader reader = new InputStreamReader(in);
            BufferedReader br = new BufferedReader(reader);
            StringBuffer sb = new StringBuffer();
            String message;
            while ((message = br.readLine()) != null) {
                sb.append(message);
            }
            System.out.println(sb);
            return sb.toString();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "";

    }

    //MMBERT predict
    public String MMBERT_predict(String name,String ques,String imagPath,String savepath,String dataset) throws IOException {

        String model = " /data2/entity/bhy/VQADEMO/model_code/MMBERT/eval.py";
        String dataset_cmd = " --data_dir " + datasetdir+dataset;
        String model_dir = " --model_dir " + savepath+'/'+name+".pt";

        String[] s=ques.split("\\s+");
        String ques2="";
        for(int i=0;i<s.length;i++){
            ques2+="_"+s[i];
        }
//        String question = " --question "+'\''+ques2+'\'';
        String question = " --question "+'"'+ques2+'"';
        String imag = " --imag "+imagPath;

        String predictcmd = torch_env + model + dataset_cmd + question + imag + model_dir;

        System.out.println(predictcmd);
        Runtime run = Runtime.getRuntime();
        try {
            Process process = run.exec(predictcmd);
            InputStream in = process.getInputStream();
            InputStreamReader reader = new InputStreamReader(in);
            BufferedReader br = new BufferedReader(reader);
            StringBuffer sb = new StringBuffer();
            String message;
            while ((message = br.readLine()) != null) {
                System.out.println(message);
                sb.append(message);
            }
            System.out.println(sb);
            return sb.toString();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "";
    }


    //ODL
    //ODL-VQA train
    public String ODL_train(String data, String name, String epoch, String batch,
                            String attention_model, String rnn, Boolean ae, Boolean maml) throws IOException {

        String dataset = " --dataset " + datasetdir + data;
        String output_cmd = " --output " + weightdir + name;
        String name_cmd = " --name " + name;
        String att_cmd = " --model " + attention_model;
        String autoencoder="";
        String maml_cmd="";
        if (ae){
            autoencoder= " --autoencoder True ";
        }
        if(maml){
            maml_cmd= " --maml True ";
        }

        String rnn_cmd = " --rnn " + rnn;
        String model = " /data2/entity/bhy/VQADEMO/model_code/ODL/main.py";
        String epoch_ = " --epochs " + epoch;
        String batch_ = " --batch_size " + batch;

        String traincmd = torch_env + model + name_cmd + dataset + output_cmd + rnn_cmd + epoch_ + batch_ + maml_cmd+
                autoencoder+att_cmd;
        System.out.println(traincmd);
        Runtime run = Runtime.getRuntime();
        try {
            Process process = run.exec(traincmd);
            InputStream in = process.getInputStream();
            InputStreamReader reader = new InputStreamReader(in);
            BufferedReader br = new BufferedReader(reader);
            StringBuffer sb = new StringBuffer();
            String message;
            while ((message = br.readLine()) != null) {
                sb.append(message);
            }
            System.out.println(sb);
            return sb.toString();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "";


    }

    //ODL-VQA predict
    public String ODL_predict(String name,String ques,String imagPath,String savepath,
                              String attention,String rnn,String ae,String maml,String data_dir) throws IOException {
        String[] s=ques.split("\\s+");
        String ques2="";
        for(int i=0;i<s.length;i++){
            ques2+="_"+s[i];
        }
        String question = " --question "+'\''+ques2+'\'';
        String imag = " --imag "+imagPath;
        String model_path=" --input "+savepath+'/'+name+".pth";
        String attention_cmd=" --model "+attention;
        String rnn_cmd=" --rnn "+rnn;
        String dataset=" --dataset "+datasetdir+data_dir;
        String autoencoder=" --autoencoder "+ae;
        String maml_cmd=" --maml "+maml;
        String model = " /data2/entity/bhy/VQADEMO/model_code/ODL/predict.py";


        String predictcmd = torch_env + model +question + imag+model_path+ attention_cmd+ rnn_cmd+ dataset+ autoencoder+ maml_cmd;
        System.out.println(predictcmd);
        Runtime run = Runtime.getRuntime();
        try {
            Process process = run.exec(predictcmd);
//            Thread.sleep(25000);
            process.waitFor();
            InputStream in = process.getInputStream();
            InputStreamReader reader = new InputStreamReader(in);
            BufferedReader br = new BufferedReader(reader);
            StringBuffer sb = new StringBuffer();
            String message;
            while ((message = br.readLine()) != null ) {
                System.out.println(message);
                System.out.println("message odl");
                sb.append(message);
            }
            in.close();

            System.out.println(sb);
            return sb.toString();
        } catch (IOException | InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "";
    }

    //AriticleNet

    //VGG-Seq2Seq train
    public String VGG_Seq2Seq_train(String name,String data, String epoch, String batch) throws IOException {
        String dataset = " --traintxt " + datasetdir + data + "/train.txt " +
                " --valtxt " + datasetdir + data + "/val.txt " +
                " --testtxt " + datasetdir + data + "/test.txt " +
                " --trainimg " + datasetdir + data + "/train " +
                " --valimg " + datasetdir + data + "/val " +
                " --testimg " + datasetdir + data + "/test ";
//        String savePath = " --model_dir " + weightdir + name;
        String data_dir = " --data_dir " + datasetdir+data;
        File file = new File(weightdir+name);
        if( !file.exists()){
            file.mkdir();
        }
        String savePath = " --model_dir " + weightdir + name+'/'+name+".pt";
        System.out.print(file.exists());
        String epoch_ = " --epochs " + epoch;
        String batch_ = " --batch-size " + batch;
        String model = " /data2/entity/bhy/VQADEMO/model_code/vgg_seq2seq/train.py";
        String traincmd = tf_env + model + data_dir+ dataset + epoch_ + batch_ + savePath;
        System.out.println(traincmd);
        Runtime run = Runtime.getRuntime();
        try {
            Process process = run.exec(traincmd);
            InputStream in = process.getInputStream();
            InputStreamReader reader = new InputStreamReader(in);
            BufferedReader br = new BufferedReader(reader);
            StringBuffer sb = new StringBuffer();
            String message;
            while ((message = br.readLine()) != null) {
                sb.append(message);
            }
            System.out.println(sb);
            return sb.toString();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "";

    }

    //VGG-Seq2Seq predict
    public String VGG_Seq2Seq_predict(String name, String data,  String imagPath, String ques,String savepath) throws IOException {

        String dataset = " --traintxt " + datasetdir + data + "/train.txt " +
                " --valtxt " + datasetdir + data + "/val.txt " +
                " --testtxt " + datasetdir + data + "/test.txt " +
                " --trainimg " + datasetdir + data + "/train " +
                " --valimg " + datasetdir + data + "/val " +
                " --testimg " + datasetdir + data + "/test ";
        //predict
        String model = " /data2/entity/bhy/VQADEMO/model_code/vgg_seq2seq/predict.py";
        String savePath = " --model_dir " + savepath + '/'+ name+".pt";
        String data_dir = " --data_dir " + datasetdir+data;

        String[] s=ques.split("\\s+");
        String ques2="";
        for(int i=0;i<s.length;i++){
            ques2+="_"+s[i];
        }
//        String question = " --test-questions \""+ques2+"\"";
        String question = " --question \"" + ques2 + "\"";
        String imag = " --imag " + imagPath;
        String predictcmd = tf_env + model + dataset + savePath + data_dir + question + imag;
        System.out.println(predictcmd);
        Runtime run = Runtime.getRuntime();
        try {
            Process process = run.exec(predictcmd);
            InputStream in = process.getInputStream();
            InputStreamReader reader = new InputStreamReader(in);
            BufferedReader br = new BufferedReader(reader);
            StringBuffer sb = new StringBuffer();
            String message;
            while ((message = br.readLine()) != null) {
                sb.append(message);
            }
            System.out.println(sb);
            return sb.toString();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "";
    }


    //NLM-VQA train
    public String NLM_train(String data, String name, String epoch, String batch,
                            String use_glove, String use_w2v, String rnn_cell) throws IOException {

        String embedding = "";
        if (use_glove == "false" && use_w2v == "true") {
            embedding += " --use-w2v -embedding-name \"PubMed-w2v.txt\"";
        } else if (use_glove == "true" && use_w2v == "false") {
            embedding += " --use-glove --embedding-name \"6B\"";
        } else {
            embedding = "";
        }
        String rnn = " --rnn-cell " + rnn_cell;
        String savePath = " --model-path " + weightdir+name;

        String dataset_ = " --data_dir "+ datasetdir+data+
                " --dataset " +datasetdir+ data + "/train_vqa_dataset.hdf5" +
                " --val-dataset " +datasetdir+ data + "/val_vqa_dataset.hdf5" +
                " --test-dataset " +datasetdir+ data + "/test_vqa_dataset.hdf5" +
                " --vocab-path "+datasetdir+ data + "/vocab_vqa.json";

        String model = " /data2/entity/bhy/VQADEMO/model_code/nlm/train_vqa.py";
        String epoch_ = " --num-epochs " + epoch;
        String batch_ = " --batch-size " + batch;
//        String n = " --run_name " + name;
        String traincmd = torch_env + model + dataset_ + embedding + rnn + epoch_ + batch_ + savePath;
        System.out.println(traincmd);
        Runtime run = Runtime.getRuntime();
        try {
            Process process = run.exec(traincmd);
            InputStream in = process.getInputStream();
            InputStreamReader reader = new InputStreamReader(in);
            BufferedReader br = new BufferedReader(reader);
            StringBuffer sb = new StringBuffer();
            String message;
            while ((message = br.readLine()) != null) {
                sb.append(message);
            }
            System.out.println(sb);
            return sb.toString();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "";


    }

    //NLM-VQA predict
    public String NLM_predict(String name,String ques,String imagPath,String dataset,String savepath,Integer batch) throws IOException {
        String model = " /data2/entity/bhy/VQADEMO/model_code/nlm/evaluate_vqa.py";

        String name_cmd = " --name " + name;
        String model_path = " --model-path " + savepath;
        String traindataset = " --dataset " + datasetdir+ dataset+"/train_vqa_dataset.hdf5";
        String valdataset = " --val-dataset " + datasetdir+ dataset+"/val_vqa_dataset.hdf5";
        String testdataset = " --test-dataset " + datasetdir+ dataset+"/test_vqa_dataset.hdf5";
//        String batchsize = " --batch-size 2" ;
        String vocab_cmd = " --vocab-path " +  datasetdir+ dataset+"/vocab_vqa.json";

        String[] s=ques.split("\\s+");
        String ques2="";
        for(int i=0;i<s.length;i++){
            ques2+="_"+s[i];
        }
        String question = " --test-questions \""+ques2+"\"";
        String imag = " --test-image "+imagPath;
        String predictcmd = torch_env + model + name_cmd + model_path + traindataset+valdataset+testdataset+
                vocab_cmd+question+imag;
        System.out.println(predictcmd);
        Runtime run = Runtime.getRuntime();
        try {
            Process process = run.exec(predictcmd);
            InputStream in = process.getInputStream();
            InputStreamReader reader = new InputStreamReader(in);
            BufferedReader br = new BufferedReader(reader);
            StringBuffer sb = new StringBuffer();
            String message;
            while ((message = br.readLine()) != null) {
                System.out.println(message);
                sb.append(message);
            }
            System.out.println(sb);
            return sb.toString();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "";
    }

    //ArticleNet
    public String AN_train(String name,String data,  String epoch, String batch) throws IOException {
//        -u /data2/entity/bhy/VQADEMO/model_code/ArticleNet/train.py \
//        --model_dir 'save/dataset.pt' \
//        --dataset "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019" \
//        --batch-size 32 \
//        --num-epoch 20
        File file = new File(weightdir+name);
        if( !file.exists()){
            file.mkdir();
        }
        String model_dir = " --model_dir " + weightdir + name+"/"+name+".pt" ;
        String model = " /data2/entity/bhy/VQADEMO/model_code/ArticleNet/train.py ";
        String epoch_ = " --num-epoch " + epoch;
        String batch_ = " --batch-size " + batch;
        String data_dir = " --dataset " + datasetdir + data;
        String traincmd = torch_env + model + model_dir + epoch_ + batch_+data_dir;
        System.out.println(traincmd);
        Runtime run = Runtime.getRuntime();
        try {
            Process process = run.exec(traincmd);
            InputStream in = process.getInputStream();
            InputStreamReader reader = new InputStreamReader(in);
            BufferedReader br = new BufferedReader(reader);
            StringBuffer sb = new StringBuffer();
            String message;
            while ((message = br.readLine()) != null) {
                sb.append(message);
            }
            System.out.println(sb);
            return sb.toString();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "";

    }

}

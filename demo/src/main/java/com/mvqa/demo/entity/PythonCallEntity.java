package com.mvqa.demo.entity;

import java.io.*;

public class PythonCallEntity {
    private final String torch_env = "python3.6";
    private final String tf_env = "/home/wxl/anaconda3/envs/tf/bin/python";
    private final String datasetdir = "/home/wxl/Documents/VQADEMO/dataset/";
    private final String weightdir = "/home/wxl/Documents/VQADEMO/weights/";
    //CGMVQA

    //MMBERT
    public String MMBERT(String mode, String data, String name, String epoch, String batch) throws IOException {

        String dataset = " --train_text_file " + datasetdir + data + "/train/train.txt" +
                " --valid_text_file " + datasetdir + data + "/valid/valid.txt" +
                " --test_text_file " + datasetdir + data + "/test/test.txt" +
                " --train_image_file " + datasetdir + data + "/train/train/" +
                " --valid_image_file " + datasetdir + data + "/valid/valid/" +
                " --test_image_file " + datasetdir + data + "/test/test/ ";

        if (mode == "train") {
            String model = " /home/wxl/Documents/VQADEMO/model_code/MMBERT/train.py";
//            String m=" --mode train";
            String epoch_ = " --epochs " + epoch;
            String batch_ = " --batch_size " + batch;
            String n = " --run_name " + name;
            String traincmd = torch_env + model + n + dataset + epoch_ + batch_;
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
//                return sb.toString();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            return "";
//            return false;
        } else {//predict
            String model = " /home/wxl/Documents/VQADEMO/model_code/MMBERT/eval.py";
//            String m=" --mode predict";
            String n = " --run_name " + name;
            String question = " --question \"what modality is shown?\"";
            String imag = " --imag ~/Documents/VQADEMO/dataset/VQA-Med-2019/ImageClef-2019-VQA-Med-Training/Train_images/synpic41148.jpg";
            String predictcmd = torch_env + model + n + question + imag + dataset;
            int rescode = 5;
            Process proc = Runtime.getRuntime().exec(predictcmd);
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while (true) {
                try {
                    rescode = proc.exitValue();
                    if (rescode != 5)
                        break;
                } catch (Exception e) {
                    return "";
                }
            }
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            return "";
        }

    }

    //ODL

    //CR

    //VGG-Seq2Seq
    public String VGG_Seq2Seq_train(String mode, String data, String name, String epoch, String batch, String save_dir) throws IOException {
        String dataset = " --train_text_file " + datasetdir + data + "/train/train.txt" +
                " --valid_text_file " + datasetdir + data + "/valid/valid.txt" +
                " --test_text_file " + datasetdir + data + "/test/test.txt" +
                " --train_image_file " + datasetdir + data + "/train/train/" +
                " --valid_image_file " + datasetdir + data + "/valid/valid/" +
                " --test_image_file " + datasetdir + data + "/test/test/ ";
        String savePath = " --save_dir " + save_dir;

        String epoch_ = " --num_epochs " + epoch;
        String batch_ = " --batch_size " + batch;
        String model = " /home/wxl/Documents/VQADEMO/model_code/VQA-Med/seq2seq_image.py";
        String m = " --mode train";
        String n = " --run_name " + name;
        String traincmd = tf_env + model + m + n + dataset + epoch_ + batch_ + savePath;
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

    public String VGG_Seq2Seq_predict(String mode, String data, String name, String imagPath, String ques) throws IOException {
        String dataset = " --train_text_file " + datasetdir + data + "/train/train.txt" +
                " --valid_text_file " + datasetdir + data + "/valid/valid.txt" +
                " --test_text_file " + datasetdir + data + "/test/test.txt" +
                " --train_image_file " + datasetdir + data + "/train/train/" +
                " --valid_image_file " + datasetdir + data + "/valid/valid/" +
                " --test_image_file " + datasetdir + data + "/test/test/ ";
        //predict
        String model = " /home/wxl/Documents/VQADEMO/model_code/VQA-Med/seq2seq_image.py";
        String m = " --mode predict";
        String n = " --run_name " + name;
        String question = " --question \"" + ques + "\"";
        String imag = " --imag " + imagPath;
        String predictcmd = tf_env + model + dataset + m + n + question + imag;
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
    public String NLM_train(String mode, String data, String name, String epoch, String batch,
                            String use_glove, String use_w2v, String rnn_cell, String save_dir) throws IOException {
//        String env="python3.6";
        String dataset = " --train_text_file " + datasetdir + data + "/train/train.txt" +
                " --valid_text_file " + datasetdir + data + "/valid/valid.txt" +
                " --test_text_file " + datasetdir + data + "/test/test.txt" +
                " --train_image_file " + datasetdir + data + "/train/train/" +
                " --valid_image_file " + datasetdir + data + "/valid/valid/" +
                " --test_image_file " + datasetdir + data + "/test/test/ ";
        String embedding = "";
        if (use_glove == "false" && use_w2v == "true") {
            embedding += " --use_w2v True";
        } else if (use_glove == "true" && use_w2v == "false") {
            embedding += " --use_glove True";
        } else {
            embedding = "";
        }
        String rnn = " --rnn_cell " + rnn_cell;
        String savePath = " --model_path " + save_dir;
        String dataset_ = " --dataset " +weightdir+ name + "/vqa_dataset.hdf5" +
                " --val_dataset " +weightdir+ name + "/vqa_dataset_val.hdf5" +
                " --train_dataset_weights " +weightdir + name + "/vqa_train_dataset_weights.json" +
                " --val_dataset_weights " +weightdir + name + "/vqa_train_dataset_weights.json" +
                " --vocab_path "+weightdir+ name + "/vocab_vqa.json";
        String model = " /home/wxl/Documents/VQADEMO/model_code/VQA/main.py";
//            String m=" --mode train";
        String epoch_ = " --num_epochs " + epoch;
        String batch_ = " --batch_size " + batch;
        String n = " --run_name " + name;
        String traincmd = torch_env + model + n + dataset + dataset_ + embedding + rnn + epoch_ + batch_ + savePath;
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
    public String NLM_predict(String name,String ques,String imagPath) throws IOException {
//        String env="python3.6";
        String weight_dir=(new File("")).getCanonicalPath()+"/weights/"+name;
        String model_path=" --model-path "+weight_dir+"/vqa-1.pkl";
        String vocab_path=" --vocab_path "+weight_dir+"/vocab_vqa.json";
        String output_path=" --output "+weight_dir;
        String model = " /home/wxl/Documents/VQADEMO/model_code/VQA/predict.py";
//            String m=" --mode predict";
        String n = " --run_name " + name;
        String question = " --question \""+ques+"\"";
        String imag = " --imagePath "+imagPath;
        String predictcmd = torch_env + model + n + question + imag+output_path+model_path+vocab_path;
        int rescode = 5;
        Process proc = Runtime.getRuntime().exec(predictcmd);
        BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
        String line = null;
        while (true) {
            try {
                rescode = proc.exitValue();
                if (rescode != 5)
                    break;
            } catch (Exception e) {
                return "false";
            }
        }
        while ((line = in.readLine()) != null) {
            System.out.println(line);
        }
        return "true";
    }


}

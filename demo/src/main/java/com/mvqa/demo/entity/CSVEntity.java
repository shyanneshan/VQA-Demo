package com.mvqa.demo.entity;

import com.mvqa.demo.model.po.ReportPo;

import java.io.*;
import java.util.ArrayList;

public class CSVEntity {

    private String filename;

    private String destpath="/home/wxl/Documents/VQADEMO/download/reports/";

    private final String[] titles={"name","classification","data","epoch","batchsize","prec","recall","f1","bleu"};

    public CSVEntity(String filename){
        this.filename=filename;
        this.destpath+=filename+".csv";
    }

    public String write2csv(ArrayList<ReportPo> reportFullVos) throws IOException {
        File outFile = new File(destpath);//写出的CSV文件

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
            String tmpString = "";
            for(int i=0;i<this.titles.length;i++){
                tmpString+=titles[i]+",";
            }
            writer.write(tmpString);
            writer.newLine();
            for(int i=0;i<reportFullVos.size();i++){
                System.out.println(reportFullVos.get(i));
                tmpString = "";
                tmpString+=reportFullVos.get(i).getName()+",";
                tmpString+=reportFullVos.get(i).getClassification()+",";
                tmpString+=reportFullVos.get(i).getData()+",";
                tmpString+=reportFullVos.get(i).getEpoch()+",";
                tmpString+=reportFullVos.get(i).getBatchsize()+",";
                tmpString+=reportFullVos.get(i).getPrec()+",";
                tmpString+=reportFullVos.get(i).getRecall()+",";
                tmpString+=reportFullVos.get(i).getF1()+",";
                tmpString+=reportFullVos.get(i).getBleu()+",";
                writer.write(tmpString);
                writer.newLine();
            }
            writer.close();
        } catch (IOException ex) {
            ex.printStackTrace();
            System.out.println("写文件出错！");
        }
        return destpath;
    }
}

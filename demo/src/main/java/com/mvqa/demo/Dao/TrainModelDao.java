package com.mvqa.demo.Dao;

import com.mvqa.demo.Mapper.ReportPoMapper;
import com.mvqa.demo.model.po.ReportPo;
import com.mvqa.demo.model.po.ReportPoExample;
import com.mvqa.demo.model.vo.ReportFullVo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;

@Repository
public class TrainModelDao {
    @Autowired
    private ReportPoMapper reportPoMapper;
    /**
     * @author     ：hybai
     * @date       ：Created in
     * @description：TODO
     */
    public ReportPo getReportPo(String modelName){
        System.out.println(modelName);
        ReportPoExample reportPoExample=new ReportPoExample();
        ReportPoExample.Criteria criteria = reportPoExample.createCriteria();
        criteria.andNameEqualTo(modelName);
        ArrayList<ReportPo> reportPos = (ArrayList<ReportPo>) reportPoMapper.selectByExample(reportPoExample);
        return reportPos.get(0);
    }

    public void deleteFailed(){
        System.out.println("delete  Failed Models in DAO.");
        ReportPoExample reportPoExample=new ReportPoExample();
        ReportPoExample.Criteria criteria = reportPoExample.createCriteria();
        criteria.andStateEqualTo("failed");
//        ArrayList<ReportPo> reportPos = (ArrayList<ReportPo>) reportPoMapper.selectByExample(reportPoExample);
        reportPoMapper.deleteByExample(reportPoExample);
    }

    /**
     * @author     ：hybai
     * @date       ：Created in
     * @description：TODO
     */
    public ArrayList<ReportPo> getAllModels(){
        ReportPoExample reportPoExample=new ReportPoExample();
        ReportPoExample.Criteria criteria = reportPoExample.createCriteria();
        criteria.andStateEqualTo("done");
        ArrayList<String> allModels=new ArrayList<String>();
        ArrayList<ReportPo> reportPos = (ArrayList<ReportPo>) reportPoMapper.selectByExample(reportPoExample);
        return reportPos;
    }

    public ArrayList<ReportPo> getAllModels_exist(){
        ReportPoExample reportPoExample=new ReportPoExample();
        ReportPoExample.Criteria criteria = reportPoExample.createCriteria();
//        criteria.andStateEqualTo("done");
        ArrayList<String> allModels=new ArrayList<String>();
        ArrayList<ReportPo> reportPos = (ArrayList<ReportPo>) reportPoMapper.selectByExample(reportPoExample);
        return reportPos;
    }

//    public ArrayList<String> getAllModels(){
//        ReportPoExample reportPoExample=new ReportPoExample();
//        ReportPoExample.Criteria criteria = reportPoExample.createCriteria();
//        criteria.andStateEqualTo("done");
//        ArrayList<String> allModels=new ArrayList<String>();
//        ArrayList<ReportPo> reportPos = (ArrayList<ReportPo>) reportPoMapper.selectByExample(reportPoExample);
//        for(int idx=0;idx<reportPos.size();idx++){
//            allModels.add(reportPos.get(idx).getName());
//            System.out.println(reportPos.get(idx));
//        }
//        return allModels;
//    }
    /**
     * @author     ：hybai
     * @date       ：Created in
     * @description：TODO
     */
    public ArrayList<ReportFullVo> getAllReport(){
        ReportPoExample reportPoExample=new ReportPoExample();
        ArrayList<ReportFullVo> reportFullVos=new ArrayList<ReportFullVo>();
        ArrayList<ReportPo> reportPos = (ArrayList<ReportPo>) reportPoMapper.selectByExample(reportPoExample);
        for(int idx=0;idx<reportPos.size();idx++){
//            if(reportPos.get(idx).getDeleted()==true){//reports deleted should not show.
//                continue;
//            }
            ReportFullVo reportFullVo =new ReportFullVo();
            reportFullVo.setData(reportPos.get(idx).getData());
            reportFullVo.setName(reportPos.get(idx).getName());
            reportFullVo.setRecall(reportPos.get(idx).getRecall());
            reportFullVo.setPrec(reportPos.get(idx).getPrec());
            reportFullVo.setF1(reportPos.get(idx).getF1());
            reportFullVo.setBleu((reportPos.get(idx).getBleu()));
            reportFullVo.setSavepath(reportPos.get(idx).getSavepath());
            reportFullVo.setEpoch(reportPos.get(idx).getEpoch());
            reportFullVo.setBatch(reportPos.get(idx).getBatchsize());
            reportFullVo.setState(reportPos.get(idx).getState());
            reportFullVo.setId(reportPos.get(idx).getId());
            reportFullVo.setClassification(reportPos.get(idx).getClassification());
//            reportFullVo.setDate(reportPos.get(idx).getSavepath());
            //process date type to string type
            DateFormat df = DateFormat.getDateInstance(DateFormat.MEDIUM, Locale.CHINA);
            String truedate=df.format(reportPos.get(idx).getDate());
//            System.out.println(truedate);
            reportFullVo.setDate(truedate);
            reportFullVos.add(reportFullVo);
        }
        return reportFullVos;
    }

    public void setTrainError(String name){
        ReportPoExample example = new ReportPoExample();
        ReportPoExample.Criteria criteria = example.createCriteria();
        criteria.andNameEqualTo(name);
        ReportPo reportPo=new ReportPo();
        reportPo.setState("failed");
        int ret=reportPoMapper.updateByExampleSelective(reportPo,example);
        if(ret==0){
            System.out.println("error! update report by name in Dao.");
        }
//        return reportPo;
    }

    /**
     * @author     ：hybai
     * @date       ：Created in
     * @description：update bleu
     */
    public ReportPo updateReportBleuByName(String name,Float bleu, String savepath){
        ReportPoExample example = new ReportPoExample();
        ReportPoExample.Criteria criteria = example.createCriteria();
        criteria.andNameEqualTo(name);
        ReportPo reportPo=new ReportPo();
        reportPo.setState("done");
        reportPo.setBleu(bleu);
        reportPo.setSavepath(savepath);
        int ret=reportPoMapper.updateByExampleSelective(reportPo,example);
        if(ret==0){
            System.out.println("error! update report by name in Dao.");
        }
        return reportPo;
    }

    /**
     * @author     ：hybai
     * @date       ：Created in
     * @description：update report by name
     */
    public ReportPo updateReportByName(String name, Integer prec, Integer recall, Integer f1, String savepath){
        ReportPoExample example = new ReportPoExample();
        ReportPoExample.Criteria criteria = example.createCriteria();
        criteria.andNameEqualTo(name);
        ReportPo reportPo=new ReportPo();
        reportPo.setState("done");
        reportPo.setF1(f1);
        reportPo.setRecall(recall);
        reportPo.setPrec(prec);
        reportPo.setSavepath(savepath);
        int ret=reportPoMapper.updateByExampleSelective(reportPo,example);
        if(ret==0){
            System.out.println("error! update report by name in Dao.");
        }
        return reportPo;


        //first select ,then build, finally update selectively
//        ReportPo reportPo=reportPoMapper.selectByPrimaryKey(id);
//        return reportPo;
    }

    /**
     * @author     ：hybai
     * @date       ：Created in
     * @description：insert to db
     */
    public ReportPo insertReport(String name,String data,String model,Integer epoch,
                                 Integer batchsize,String rnn_cell,String embedding,
                                 String attention, String constructor) throws ParseException {
        //first select ,then build, finally update selectively
        ReportPo reportPo=new ReportPo();
        reportPo.setState("running");
        reportPo.setName(name);
        reportPo.setEpoch(epoch);
        reportPo.setBatchsize(batchsize);
        reportPo.setRnnCell(rnn_cell);
        reportPo.setEmbedding(embedding);
        reportPo.setAttention(attention);
        reportPo.setConstructor(constructor);
        reportPo.setEmbedding(embedding);
        reportPo.setData(data);
        ZoneId zoneId = ZoneId.systemDefault();
        LocalDate localDate = LocalDate.now();
        ZonedDateTime zdt = localDate.atStartOfDay(zoneId);
        Date date = Date.from(zdt.toInstant());
        DateFormat df = DateFormat.getDateInstance(DateFormat.MEDIUM, Locale.CHINA);
        String truedate=df.format(date);
        System.out.println(truedate);
        SimpleDateFormat sdf=new SimpleDateFormat("yyyy-MM-dd");//小写的mm表示的是分钟
        Date d=sdf.parse(truedate);
        reportPo.setDate(d);
        reportPo.setClassification(model);
        int ret =reportPoMapper.insertSelective(reportPo);
        if(ret==0){
            System.out.println("error! insert report in Dao.");
        }
        return reportPo;
    }

}

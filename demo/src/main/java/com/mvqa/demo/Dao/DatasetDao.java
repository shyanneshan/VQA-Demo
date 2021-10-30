package com.mvqa.demo.Dao;

import com.mvqa.demo.Mapper.DatasetPoMapper;
import com.mvqa.demo.model.po.DatasetPo;
import com.mvqa.demo.model.po.DatasetPoExample;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;

@Repository
public class DatasetDao {

    @Autowired
    private DatasetPoMapper datasetPoMapper;
    /**
     * @author     ：hybai
     * @date       ：Created in
     * @description：TODO
     */
    public ArrayList<DatasetPo> getDatasetPo(){

        DatasetPoExample datasetPoExample=new DatasetPoExample();
        DatasetPoExample.Criteria criteria = datasetPoExample.createCriteria();
        ArrayList<DatasetPo> datasetPos = (ArrayList<DatasetPo>) datasetPoMapper.selectByExample(datasetPoExample);
        for(int i=0;i<datasetPos.size();i++){
            System.out.println(datasetPos.get(i).getName());
        }
        return datasetPos;
    }

    public void addDataset(DatasetPo datasetPo){
        datasetPoMapper.insert(datasetPo);

    }

}

package com.mvqa.demo.entity;

import com.mvqa.demo.model.po.ctInfoPo;
import com.mvqa.demo.model.po.ctValPo;
import lombok.Data;

@Data
public class Photo {
    private Long id;
    private String patientId;
    private String sym;
    private String photoId;
    private String diaList;
    private String annotation;
    private String description;
    private String boneName;
    private String direction;
    private String dataset;

    public String getDataset() {
        return dataset;
    }

    public void setDataset(String dataset) {
        this.dataset = dataset;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getBoneName() {
        return boneName;
    }

    public void setBoneName(String boneName) {
        this.boneName = boneName;
    }

    public String getDirection() {
        return direction;
    }

    public void setDirection(String direction) {
        this.direction = direction;
    }


    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getPatientId() {
        return patientId;
    }

    public void setPatientId(String patientId) {
        this.patientId = patientId;
    }

    public String getSym() {
        return sym;
    }

    public void setSym(String sym) {
        this.sym = sym;
    }

    public String getPhotoId() {
        return photoId;
    }

    public void setPhotoId(String photoId) {
        this.photoId = photoId;
    }

    public String getDiaList() {
        return diaList;
    }

    public void setDiaList(String diaList) {
        this.diaList = diaList;
    }

    public String getAnnotation() {
        return annotation;
    }

    public void setAnnotation(String annotation) {
        this.annotation = annotation;
    }

    public String getFlag() {
        return flag;
    }

    public void setFlag(String flag) {
        this.flag = flag;
    }

    private String flag;
    private String type;

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getPosition() {
        return position;
    }

    public void setPosition(String position) {
        this.position = position;
    }

    private String position;

    public Photo(){

    }

    public Photo(ctInfoPo ctInfoPo){
        setDataset(ctInfoPo.getDataset());
        setId(Long.parseLong(String.valueOf(ctInfoPo.getId())));
        setPhotoId(ctInfoPo.getPhotoId());
        setPatientId(ctInfoPo.getPatientId());
        setSym(ctInfoPo.getSym());
        setFlag(ctInfoPo.getStatus());
        setDiaList(ctInfoPo.getDiaList());
        setAnnotation(ctInfoPo.getAnnotation());
    }

    public ctInfoPo createCtInfoPo(){
        ctInfoPo ctInfoPo=new ctInfoPo();
        ctInfoPo.setDataset(this.getDataset());
        ctInfoPo.setPhotoId(this.getPhotoId());
//        ctInfoPo.setId(Integer.parseInt(String.valueOf(this.getId())));
        ctInfoPo.setDiaList(this.getDiaList());
        ctInfoPo.setPatientId(this.getPatientId());
        return ctInfoPo;
    }

    public ctValPo createCtValPo(){
        ctValPo ctValPo=new ctValPo();
        ctValPo.setDataset(this.getDataset());
        ctValPo.setPosition(this.getPosition());
        ctValPo.setBoneName(this.getBoneName());
        ctValPo.setPhotoId(this.getPhotoId());
        ctValPo.setDescription(this.getDescription());
        ctValPo.setId(this.getId());
        ctValPo.setDiaList(this.getDiaList());
        ctValPo.setPatientId(this.getPatientId());
        ctValPo.setType(this.getType());
        ctValPo.setDirection(this.getDirection());
        return ctValPo;
    }
}

package com.mvqa.demo.entity;

import lombok.Data;

/**
 * Medical Archive Entity
 * @author XYX
 */
@Data
public class MedicalArchiveEntity {
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getUserId() {
        return userId;
    }

    public void setUserId(Long userId) {
        this.userId = userId;
    }


//    public String getTitle() {
//        return title;
//    }
//
//    public void setTitle(String title) {
//        this.title = title;
//    }
//
    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public boolean isStatus() {
        return status;
    }

    public void setStatus(boolean status) {
        this.status = status;
    }

    public String getFilePath() {
        return FilePath;
    }

    public void setFilePath(String filePath) {
        FilePath = filePath;
    }

    public String getTxtFilePath() {
        return txtFilePath;
    }

    public void setTxtFilePath(String txtFilePath) {
        this.txtFilePath = txtFilePath;
    }

    private Long id;
    private Long userId;
//    private String title;
    private String description;
    private boolean status;
    private String FilePath;
    private String txtFilePath;
}

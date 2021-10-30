//package com.mvqa.demo.Service;
//
//import com.mvqa.demo.Controller.MedicalArchiveDao;
//import com.java.adds.entity.MedicalArchiveEntity;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.stereotype.Service;
//
//import java.util.ArrayList;
//
///**
// * Medical Archive Service
// * @author XYX
// */
//@Service
//public class MedicalArchiveService {
//
//    @Autowired
//    private MedicalArchiveDao medicalArchiveDao;
//
//    /**
//     * Get Medical Archive By User's Id
//     * @param userId user's id
//     * @return A KG ArrayList
//     */
//    public ArrayList<MedicalArchiveEntity> getMedicalArchiveByUserId(Long userId) {
//        return medicalArchiveDao.getMedicalArchiveByUserId(userId);
//    }
//
//    /**
//     * Upload Medical Archive
//     * @param medicalArchive medical archive
//     * @return medical archive id
//     */
//    public Long uploadMedicalArchive(MedicalArchiveEntity medicalArchive) {
//        Long medicalArchiveId = medicalArchiveDao.uploadMedicalArchiveByUserId(medicalArchive);
//        if (medicalArchiveId >= 0) {
//            return medicalArchiveId;
//        } else {
//            return -1L;
//        }
//    }
//
//    /**
//     * Update Medical Archive
//     * @param medicalArchive medical archive
//     * @return medical archive id
//     */
//    public Long updateMedicalArchive(MedicalArchiveEntity medicalArchive) {
//        Long medicalArchiveId = medicalArchiveDao.updateMedicalArchive(medicalArchive);
//        if (medicalArchiveId >= 0) {
//            return medicalArchiveId;
//        } else {
//            return -1L;
//        }
//    }
//
//    /**
//     * Get Medical Archive By Id
//     * @param archiveId medical archive id
//     * @return medical archive id
//     */
//    public MedicalArchiveEntity getMedicalArchiveById(Long archiveId) {
//        return medicalArchiveDao.getMedicalArchiveById(archiveId);
//    }
//}

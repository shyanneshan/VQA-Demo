-- MySQL dump 10.13  Distrib 8.0.25, for Linux (x86_64)
--
-- Host: 127.0.0.1    Database: vqademo
-- ------------------------------------------------------
-- Server version	8.0.25-0ubuntu0.20.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ct_information`
--

DROP TABLE IF EXISTS `ct_information`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ct_information` (
  `id` int NOT NULL AUTO_INCREMENT,
  `patient_id` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `sym` varchar(1000) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `photo_id` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `dia_list` varchar(1000) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `annotation` varchar(1000) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `status` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `dataset` varchar(30) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb3 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ct_information`
--

LOCK TABLES `ct_information` WRITE;
/*!40000 ALTER TABLE `ct_information` DISABLE KEYS */;
INSERT INTO `ct_information` VALUES (1,'1','外伤致左肘部疼痛、畸形、活动障碍3天。外院予膏药外敷对症处理，因患肢肿痛明显，关节活动障碍，就诊于当地医院拍片检查提示：左尺骨鹰嘴骨折，为求进一步治疗转诊我院，拟诊\"左尺骨鹰嘴骨折\"收入科，自发病来，精神差，饮食好，大便、小便正常。','11','左尺骨鹰嘴骨折','外院X线片示（未见报告单）：左尺骨鹰嘴骨折','0','vqademo'),(2,'1','外伤致左肘部疼痛、畸形、活动障碍3天。外院予膏药外敷对症处理，因患肢肿痛明显，关节活动障碍，就诊于当地医院拍片检查提示：左尺骨鹰嘴骨折，为求进一步治疗转诊我院，拟诊\"左尺骨鹰嘴骨折\"收入科，自发病来，精神差，饮食好，大便、小便正常。','12','左尺骨鹰嘴骨折','外院X线片示（未见报告单）：左尺骨鹰嘴骨折','0','vqademo'),(3,'1','外伤致左肘部疼痛、畸形、活动障碍3天。外院予膏药外敷对症处理，因患肢肿痛明显，关节活动障碍，就诊于当地医院拍片检查提示：左尺骨鹰嘴骨折，为求进一步治疗转诊我院，拟诊\"左尺骨鹰嘴骨折\"收入科，自发病来，精神差，饮食好，大便、小便正常。','13','左尺骨鹰嘴骨折','外院X线片示（未见报告单）：左尺骨鹰嘴骨折','0','vqademo'),(4,'2','硬物打伤头、颜面及胸腹等部位，伤后头痛、头晕伴出血不止，多根牙断裂伴疼痛不适，右手肿痛明显，给予止痛、营养神经等治疗后拟“外伤性头痛、头皮裂伤、外伤性牙折、全身多处软组织挫擦伤、右手第2掌骨骨折？”收入我科继续观察治疗。','21','外伤性头痛,头皮裂伤,右第2掌骨基底部骨折,全身多处软组织挫擦伤,创伤性鼓膜穿孔（左）,右腕小多角骨骨折','辅助检查：（2018-07-15，我院急诊）头胸CT：1、颅内未见明显血肿，颅骨未见骨折。2、考虑左下肺胸膜下炎性增生性小结节，建议随访；双侧肋骨未见明显错位骨折，必要时建议三维重建。','0','vqademo'),(5,'2','硬物打伤头、颜面及胸腹等部位，伤后头痛、头晕伴出血不止，多根牙断裂伴疼痛不适，右手肿痛明显，给予止痛、营养神经等治疗后拟“外伤性头痛、头皮裂伤、外伤性牙折、全身多处软组织挫擦伤、右手第2掌骨骨折？”收入我科继续观察治疗。','22','右第2掌骨基底部骨折,右腕小多角骨骨折','辅助检查：（2018-07-15，我院急诊）头胸CT：1、颅内未见明显血肿，颅骨未见骨折。2、考虑左下肺胸膜下炎性增生性小结节，建议随访；双侧肋骨未见明显错位骨折，必要时建议三维重建。','0','vqademo'),(6,'2','硬物打伤头、颜面及胸腹等部位，伤后头痛、头晕伴出血不止，多根牙断裂伴疼痛不适，右手肿痛明显，给予止痛、营养神经等治疗后拟“外伤性头痛、头皮裂伤、外伤性牙折、全身多处软组织挫擦伤、右手第2掌骨骨折？”收入我科继续观察治疗。','23','外伤性头痛,头皮裂伤,右第2掌骨基底部骨折,全身多处软组织挫擦伤,创伤性鼓膜穿孔（左）,右腕小多角骨骨折','辅助检查：（2018-07-15，我院急诊）头胸CT：1、颅内未见明显血肿，颅骨未见骨折。2、考虑左下肺胸膜下炎性增生性小结节，建议随访；双侧肋骨未见明显错位骨折，必要时建议三维重建。','0','vqademo');
/*!40000 ALTER TABLE `ct_information` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ct_validation`
--

DROP TABLE IF EXISTS `ct_validation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ct_validation` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `patient_id` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `photo_id` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `dia_list` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `description` varchar(1000) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `bone_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `direction` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `type` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `position` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci DEFAULT NULL,
  `dataset` varchar(30) DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb3 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ct_validation`
--

LOCK TABLES `ct_validation` WRITE;
/*!40000 ALTER TABLE `ct_validation` DISABLE KEYS */;
INSERT INTO `ct_validation` VALUES (1,'1','11','左尺骨鹰嘴骨折','左尺骨鹰嘴骨折','4','2','2','3','vqademo'),(2,'1','11','左尺骨鹰嘴骨折','患肢肿痛明显，关节活动障碍','4','2','2','3','vqademo'),(3,'2','22','外伤性头痛,头皮裂伤,右第2掌骨基底部骨折,全身多处软组织挫擦伤,创伤性鼓膜穿孔（左）,右腕小多角骨骨折','双侧肋骨未见明显错位骨折，必要时建议三维重建。','3','3','1','5','vqademo'),(4,'1','11','左尺骨鹰嘴骨折','患肢肿痛明显，关节活动障碍','4','2','2','3','vqademo'),(5,'1','11','左尺骨鹰嘴骨折','患肢肿痛明显，关节活动障碍','4','2','2','3','vqademo'),(6,'2','22','右第2掌骨基底部骨折,右腕小多角骨骨折','多根牙断裂伴疼痛不适，右手肿痛明显','3','3','1','1','vqademo');
/*!40000 ALTER TABLE `ct_validation` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `dataset`
--

DROP TABLE IF EXISTS `dataset`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `dataset` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(30) NOT NULL,
  `description` varchar(60) DEFAULT NULL,
  `status` int NOT NULL DEFAULT '0',
  `link` varchar(300) DEFAULT NULL,
  `islabeled` varchar(4) DEFAULT NULL,
  `test` int DEFAULT NULL,
  `valid` int DEFAULT NULL,
  `train` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `dataset_id_uindex` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=20 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `dataset`
--

LOCK TABLES `dataset` WRITE;
/*!40000 ALTER TABLE `dataset` DISABLE KEYS */;
INSERT INTO `dataset` VALUES (1,'VQA-Med-2019','VQA-RAD dataset',1,'vqa-dataset.obs.cn-north-4.myhuaweicloud.com:443/VQA-Med-2019.zip?AccessKeyId=N6RORRHWKMOULERQYSPF&Expires=1654687886&Signature=rGLtwj0kRuOxPfL%2BJLAMoBF9IQE%3D','1',NULL,NULL,NULL),(2,'VQA-RAD','ImageCLEF 2019',1,'vqa-dataset.obs.cn-north-4.myhuaweicloud.com:443/VQA-RAD.rar?AccessKeyId=N6RORRHWKMOULERQYSPF&Expires=1654687968&Signature=R207OXUpMQwI9RRRQzaFnVcOshk%3D','1',NULL,NULL,NULL),(19,'vqademo','for illustration',1,NULL,'0',2,2,6);
/*!40000 ALTER TABLE `dataset` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `report`
--

DROP TABLE IF EXISTS `report`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `report` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(20) DEFAULT NULL,
  `date` date DEFAULT NULL,
  `state` varchar(10) DEFAULT NULL,
  `data` varchar(20) DEFAULT NULL,
  `savepath` varchar(256) DEFAULT NULL,
  `prec` int DEFAULT NULL,
  `recall` int DEFAULT NULL,
  `f1` int DEFAULT NULL,
  `classification` varchar(50) DEFAULT NULL,
  `epoch` int DEFAULT NULL,
  `batchsize` int DEFAULT NULL,
  `rnn_cell` varchar(20) DEFAULT NULL,
  `embedding` varchar(20) DEFAULT NULL,
  `bleu` float(5,2) DEFAULT NULL,
  `deleted` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `report`
--

LOCK TABLES `report` WRITE;
/*!40000 ALTER TABLE `report` DISABLE KEYS */;
INSERT INTO `report` VALUES (1,'nlm1','2021-06-13','done','VQA-Med-2019','/home/wxl/Documents/VQADEMO/weights/nlm1',NULL,NULL,NULL,'NLM',10,64,'GRU','none',0.12,NULL),(2,'vgg1','2021-06-13','done','VQA-RAD','/home/wxl/Documents/VQADEMO/weights/vgg1',NULL,NULL,NULL,'VGG-Seq2Seq',100,32,'','',0.41,NULL),(3,'mmbert1','2021-06-13','done','VQA-Med-2019','/home/wxl/Documents/VQADEMO/weights/mmbert1',6355,5312,6478,'MMBERT',60,32,'','',NULL,NULL),(5,'nlm2','2021-06-13','done','VQA-RAD',NULL,NULL,NULL,NULL,'NLM',10,16,'LSTM','none',0.00,NULL);
/*!40000 ALTER TABLE `report` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2021-06-14 16:02:43

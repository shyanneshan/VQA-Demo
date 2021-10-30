# models Code

**This is the open source code of our paper.**

#### Target

We build a VQA application platform from dataset generation to model application.

#### Video

We upload the recorded video, and users can learn how to use the system according to the [**videos**](https://github.com/shyanneshan/VQA-Demo/tree/master/video).

#### Module

The demo has the following modules.

##### Dataset generation

Upload the ZIP file of the medical record and build the dataset.

![](pic/dataset1.png)

![](pic/dataset2.png)

##### Label

Label by patient id and picture id.

![](pic/label1.png)

![](pic/label2.png)

##### Model training

![](pic/practice.png)

Medical VQA model is provided, and users can select parameter annotation. (For the time being, there are only three models: NLM, VGG-Seq2Seq, MMbert, and models will be introduced in the future.)

##### Model application

After the training, the model score report is provided and the model effect can be tested on the AI Robot interface.

![](pic/report.png)

![](pic/ai.png)

#### The environment

> Ubuntu 20.04
>
> mysql 8.0.25
>
> jdk 1.8
>
> Vue

#### Build

**How to run frontend?**

The front files are in the [**fronted folder.**](https://github.com/shyanneshan/VQA-Demo/tree/master/fronted)

```
npm install
npm run dev
```

**How to run backend?**
The back-end code is in the [**demo folder.**](https://github.com/shyanneshan/VQA-Demo/tree/master/demo)

**Configuration**

The sql file is in the [resources](https://github.com/shyanneshan/VQA-Demo/tree/master/demo/src/main/resources). We only provide several pieces of records for illustration.
We've created three new Python environments. Python environments are provided by [requirements](https://github.com/shyanneshan/VQA-Demo/tree/master/demo/src/main/resources/python).


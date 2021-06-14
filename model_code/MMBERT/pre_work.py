import process_tocsv
import pandas as pd

def getDfWithMode(args):
    l=[]
    testdf= process_tocsv.getTestCsv(args)
    traindf = process_tocsv.getTrainCsv(args)
    validdf = process_tocsv.getValidCsv(args)
    # testdf=[testdf]
    # print(testdf)
    # df=utils.make_df(r'C:\Users\PC\Desktop\testpath',args)
    # print(df)
    l.append(testdf)
    l.append(traindf)
    l.append(validdf)
    all_df=pd.concat(l)
    # print(all_df)
    return all_df

# if __name__=="__main__":
#     l=[]
#     testdf=process_tocsv.getTestCsv(r'C:\Users\PC\Documents\GitHub\VQA-Med-2019\VQAMed2019Test\VQAMed2019_Test_Questions_w_Ref_Answers.txt')
#     # testdf=[testdf]
#     # print(testdf)
#     df=utils.make_df(r'C:\Users\PC\Desktop\testpath')
#     # print(df)
#     l.append(testdf)
#     l.append(df)
#     all_df=pd.concat(l)
#     print(all_df)
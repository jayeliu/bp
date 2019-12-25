#include <iostream>
#include <time.h>

#include "NetWork.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "dataLoader.h"
#include "Utils.h"

/** indicate 0 ~ 9 */
#define NUM_NET_OUT 10
#define NUM_HIDDEN 500
#define NET_LEARNING_RATE 0.5
#define CHUNK_SIZE 200
#define PIC_SIZE 785

#define TRAIN_IMAGES_URL "../data/train-images.idx3-ubyte"
#define TRAIN_LABELS_URL "../data/train-labels.idx1-ubyte"

#define TEST_IMANGES_URL "../data/t10k-images.idx3-ubyte"
#define TEST_LABELS_URL  "../data/t10k-labels.idx1-ubyte"


void showNumber(unsigned char pic[], int width, int height) {
    int idx = 0;
    for (int i=0; i < height; i++) {
        for (int j = 0; j < width; j++ ) {

            if (pic[idx++]) {
                cout << "1";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }
}

inline void preProcessInputData(const unsigned char src[], double out[], int size) {//将输入的数据2值化
    for (int i = 0; i < size; i++) {
        out[i] = (src[i] >= 128) ? 1.0 : 0.0;
    }
}



double trainEpoch(dataLoader& src, NetWork& bpnn, int imageSize, int numImages) {
    double net_target[NUM_NET_OUT];
    char* temp = new char[imageSize];

    double* net_train = new double[imageSize];
    for (int i = 0; i < numImages; i++) {
        int label = 0;
        memset(net_target, 0, NUM_NET_OUT * sizeof(double));

        if (src.read(&label, temp)) {
            net_target[label] = 1.0;
            preProcessInputData((unsigned char*)temp, net_train, imageSize);
            bpnn.training(net_train, net_target);
        }
        else {
            cout << "读取训练数据失败" << endl;
            break;
        }
    }

    // cout << "the error is:" << bpnn.getError() << " after training " << endl;

    delete []net_train;
    delete []temp;

    return bpnn.getError();
}

int testRecognition(dataLoader& testData, NetWork& bpnn, int imageSize, int numImages) {
    int ok_cnt = 0;
    double* net_out = NULL;
    char* temp = new char[imageSize];
    double* net_test = new double[imageSize];
    for (int i = 0; i < numImages; i++) {
        int label = 0;
        cout<<"已测试:"<<i<<"\r";
        if (testData.read(&label, temp)) {//读取图片和label
            preProcessInputData((unsigned char*)temp, net_test, imageSize);
            bpnn.process(net_test, &net_out);

            int idx = -1;
            double max_value = -99999;
            for (int i = 0; i < NUM_NET_OUT; i++) {//找输出数组最大的可能性
                if (net_out[i] > max_value) {
                    max_value = net_out[i];
                    idx = i;
                }
            }

            if (idx == label) {//如果与该图片的label相同则正确数加一
                ok_cnt++;
            }

        }
        else {
            cout << "read test data failed" << endl;
            break;
        }
    }
    delete []net_test;
    delete []temp;
    return ok_cnt;
}


int main(int argc, char* argv[]) {
    int task_count = 0;
    int rank = 0;
    int lenth = 0;
    int tag = 0;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;
    int ret = MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &task_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name,&namelen);
    if(task_count<=1)
        {
            printf("the progress number should be more than 1\n");
            MPI_Finalize();
            return -1;
        }
    printf("Process %d of %d is on %s\n",rank, task_count, processor_name);//output the information of process



    dataLoader src;
    dataLoader testData;
    NetWork* bpnn = NULL;
    srand((int)time(0));


    if (src.openImageFile(TRAIN_IMAGES_URL) && src.openLabelFile(TRAIN_LABELS_URL)) {
        int imageSize = src.imageLength();
        int numImages = src.numImage();
        bpnn = new NetWork(imageSize, NET_LEARNING_RATE);
        // 加入隐藏层
        bpnn->addNeuronLayer(NUM_HIDDEN);//在隐藏层有100个神经细胞
        // 加入输出层
        bpnn->addNeuronLayer(NUM_NET_OUT);//10个输出
        if(rank==0)
        cout << "开始进行训练：" << endl;
        uint64_t st = timeNowMs();

        int turn = numImages/((task_count-1)*CHUNK_SIZE);//cycle index
        int turn0=turn;
        int over=numImages%((task_count-1)*CHUNK_SIZE);
        int oversize=over/task_count;//every process need to do in last turn
        int overover=over%task_count;//which process needs to do more
        if(over!=0)
            turn0=turn+1;
        for (int i = 0; i < turn0; ++i)
        {
            if(rank==0)
                cout << "已学习：" << i*(task_count-1)*CHUNK_SIZE << "\r";

            if(rank!=0)
            {
            	int	size=CHUNK_SIZE;
            	int offset=0;
            	if(i!=turn)
                {
                	offset=i*(task_count-1)*CHUNK_SIZE+(rank-1)*CHUNK_SIZE;//计算偏移量
                }
                else
                {
                	if((rank-1)<overover)
                		{
                			size=oversize+1;
                			offset=i*(task_count-1)*CHUNK_SIZE+(rank-1)*oversize+(rank-1);
                			}
                	else 
                	{
                		size=oversize;
                		offset=i*(task_count-1)*CHUNK_SIZE+(rank-1)*oversize+overover;
                		}
                }
                
                
                
                
                src.movepoint(offset);//移动文件的文件指针
                double err = trainEpoch(src, *bpnn, imageSize, size);//进行训练
                
                
                
                
                
                //每个进程将训练完的参数发送给0进程进行参数平均的运算
                for (int m = 0; m < NUM_HIDDEN; ++m)
                {
                    MPI_Send(bpnn->mNeuronLayers[0]->mWeights[m], PIC_SIZE , MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
                }
                for (int m = 0; m < NUM_NET_OUT; ++m)
                {
                    MPI_Send(bpnn->mNeuronLayers[1]->mWeights[m],NUM_HIDDEN+ 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
                }
                for (int m = 0; m < NUM_HIDDEN; ++m)
                {
                    MPI_Recv(bpnn->mNeuronLayers[0]->mWeights[m],PIC_SIZE , MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
                }
                for (int m = 0; m < NUM_NET_OUT; ++m)
                {
                    MPI_Recv(bpnn->mNeuronLayers[1]->mWeights[m], NUM_HIDDEN+1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
                }
            }


            else
            {
                double *fWeight=new double[PIC_SIZE];
                double *oWeight=new double[NUM_HIDDEN+1];
                double **sumfWeight=new double*[NUM_HIDDEN];
                //0进程接受每个进程发送的数据，并进行平均化再发送出去
                for (int m = 0; m < NUM_HIDDEN; ++m)
                {
                    sumfWeight[m]=new double[PIC_SIZE];
                    memset(sumfWeight[m],0,PIC_SIZE*sizeof(double));
                }
                double **sumoWeight=new double*[NUM_NET_OUT];
                for (int m = 0; m < NUM_NET_OUT; ++m)
                {
                    sumoWeight[m]=new double[NUM_HIDDEN+1];
                    memset(sumoWeight[m],0,(NUM_HIDDEN+1)*sizeof(double));
                }
				//接收
                for (int m = 1; m < task_count; ++m)
                {
                    for (int j = 0; j < NUM_HIDDEN; ++j)
                    {
                        MPI_Recv(fWeight,PIC_SIZE , MPI_DOUBLE, m, tag, MPI_COMM_WORLD, &status);
                        for (int k = 0; k < PIC_SIZE; ++k)
                        {
                            sumfWeight[j][k]+=fWeight[k];
                        }
                    }

                    for (int j = 0; j < NUM_NET_OUT; ++j)
                    {
                        MPI_Recv(oWeight, (NUM_HIDDEN+1), MPI_DOUBLE, m, tag, MPI_COMM_WORLD, &status);
                        for (int k = 0; k < (NUM_HIDDEN+1); ++k)
                        {
                            sumoWeight[j][k]+=oWeight[k];
                        }
                    }
                }


				//平均
                for (int m = 0; m < NUM_HIDDEN; ++m)
                {
                    for (int j = 0; j < PIC_SIZE; ++j)
                    {
                        sumfWeight[m][j]/=(task_count-1);
                    }
                }
                for (int m = 0; m < NUM_NET_OUT; ++m)
                {
                    for (int j = 0; j < (NUM_HIDDEN+1); ++j)
                    {
                        sumoWeight[m][j]/=(task_count-1);
                    }
                }



				//发送
                for (int m = 1; m < task_count; ++m)
                {
                    for (int j = 0; j < NUM_HIDDEN; ++j)
                    {
                        MPI_Send(sumfWeight[j], PIC_SIZE , MPI_DOUBLE, m, tag, MPI_COMM_WORLD);
                    }
                    for (int j = 0; j < NUM_NET_OUT; ++j)
                    {
                        MPI_Send(sumoWeight[j],NUM_HIDDEN+ 1, MPI_DOUBLE, m, tag, MPI_COMM_WORLD);
                    }
                }
                
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        src.reset();//重置dataloader变量
        if(rank==0)
        {   
            cout << "训练结束，花费时间: " << (timeNowMs() - st)/1000 << "秒" << endl;
        }
        st = timeNowMs();
        
        if (testData.openImageFile(TEST_IMANGES_URL) && testData.openLabelFile(TEST_LABELS_URL)) {
            imageSize = testData.imageLength();
            numImages = testData.numImage();
            if(rank==0)
            cout << "开始进行测试：" << endl;
			if(rank==1)
            {
            	int ok_cnt = testRecognition(testData, *bpnn, imageSize, numImages);//测试图片
            	if(rank!=0)
            	cout << "测试结束，花费时间："
                << (timeNowMs() - st)/1000 << "秒, " 
                <<  "成功比例: " << ok_cnt/(double)numImages*100 << "%" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
        }
        else {
            cout << "打开测试文件失败" << endl;
        }


    }
    else {
        cout << "open train image file failed" << endl;
    }

    if (bpnn) {
        delete bpnn;
    }
    getchar();

    return 0;
}

// face_recognition.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "Process.h"
#include "My_Matrix.h"

int _tmain(int argc, _TCHAR* argv[])
{
	double *T,*L,*m,*b,*q,*c,*p_q,*projected_train,*T_test,*projected_test,*eigenvector,*Euc_dist;
	double eps,temp;
	int i,j,flag,iteration,num_q;
	char res[20];
	IplImage *tmp_img,*test_img;

	T = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*TRAIN_NUM);	//原始数据
	T_test = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*1);		//测试数据
	m = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH);		//平均值
	L = (double *)malloc(sizeof(double)*TRAIN_NUM*TRAIN_NUM);		//L=T'*T，协方差矩阵
	b = (double *)malloc(sizeof(double)*TRAIN_NUM);				//L的特征值
	q = (double *)malloc(sizeof(double)*TRAIN_NUM*TRAIN_NUM);	//L特征值对应的特征向量
	c = (double *)malloc(sizeof(double)*TRAIN_NUM);				//实对称三对角矩阵的次对角线元素

	eps = 0.000001;
	memset(L,0,sizeof(double)*TRAIN_NUM*TRAIN_NUM);
	
	//存储图像数据到T矩阵
	for (i=1;i<=TRAIN_NUM;i++)
	{
		sprintf(res,".\\TrainDatabase\\%d.jpg",i);
		tmp_img = cvLoadImage(res,CV_LOAD_IMAGE_GRAYSCALE);
		load_data(T,tmp_img,i);
	}
	
	//求T矩阵行的平均值
	calc_mean(T,m);

	//构造协方差矩阵
	calc_covariance_matrix(T,L,m);

	//求L的特征值，特征向量
	iteration = 60;
	cstrq(L,TRAIN_NUM,q,b,c);
	flag = csstq(TRAIN_NUM,b,c,q,eps,iteration); //数组q中第j列为数组b中第j个特征值对应的特征向量
	if (flag<0)
	{
		printf("fucking failed!\n");
	}else
	{
		printf("success to get eigen value and vector\n");
	}

	//对L挑选合适的特征值，过滤特征向量
	num_q=0;
	for (i=0;i<TRAIN_NUM;i++)
	{
		if (b[i]>1)
		{
			num_q++;
		}
	}
	p_q = (double *)malloc(sizeof(double)*TRAIN_NUM*TRAIN_NUM);			//挑选后的L的特征向量，仅过滤，未排序
	projected_train = (double *)malloc(sizeof(double)*TRAIN_NUM*num_q);	//投影后的训练样本特征空间
	eigenvector = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*num_q);//Pe=λe,Q(Xe)=λ(Xe)，投影变换向量
	pick_eignevalue(b,q,p_q,num_q);
	get_eigenface(p_q,T,num_q,projected_train,eigenvector);

	//读取测试图像
	test_img = cvLoadImage(".\\TestDatabase\\4.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	projected_test = (double *)malloc(sizeof(double)*num_q*1);//在特征空间投影后的测试样本
	for (i=0;i<IMG_HEIGHT;i++)
	{
		for (j=0;j<IMG_WIDTH;j++)
		{
			T_test[i*IMG_WIDTH+j] = (double)(unsigned char)test_img->imageData[i*IMG_WIDTH+j] - m[i*IMG_WIDTH+j];
		}
	}

	//将待测数据投影到特征空间
	memset(projected_test,0,sizeof(double)*num_q);
	matrix_mutil(projected_test,eigenvector,T_test,num_q,IMG_WIDTH*IMG_HEIGHT,1);

	//计算projected_test与projected_train中每个向量的欧氏距离
	Euc_dist = (double *)malloc(sizeof(double)*TRAIN_NUM);
	for (i=0;i<TRAIN_NUM;i++)
	{
		temp = 0;
		for (j=0;j<num_q;j++)
		{
			temp = temp + (projected_test[j]-projected_train[j*TRAIN_NUM+i])*(projected_test[j]-projected_train[j*TRAIN_NUM+i]);
		}
		Euc_dist[i] = temp;
		//printf("%f \n",temp);
	}
	//寻找最小距离
	double min = Euc_dist[0];
	int label;
	for (i=0;i<TRAIN_NUM;i++)
	{
		if (min>=Euc_dist[i])
		{
			min = Euc_dist[i];
			label = i;
		}
	}
	printf("%d.jpg is mathcing!",label+1);
	return 0;
}












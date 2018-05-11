// face_recognition.cpp : �������̨Ӧ�ó������ڵ㡣
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

	T = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*TRAIN_NUM);	//ԭʼ����
	T_test = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*1);		//��������
	m = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH);		//ƽ��ֵ
	L = (double *)malloc(sizeof(double)*TRAIN_NUM*TRAIN_NUM);		//L=T'*T��Э�������
	b = (double *)malloc(sizeof(double)*TRAIN_NUM);				//L������ֵ
	q = (double *)malloc(sizeof(double)*TRAIN_NUM*TRAIN_NUM);	//L����ֵ��Ӧ����������
	c = (double *)malloc(sizeof(double)*TRAIN_NUM);				//ʵ�Գ����ԽǾ���ĴζԽ���Ԫ��

	eps = 0.000001;
	memset(L,0,sizeof(double)*TRAIN_NUM*TRAIN_NUM);
	
	//�洢ͼ�����ݵ�T����
	for (i=1;i<=TRAIN_NUM;i++)
	{
		sprintf(res,".\\TrainDatabase\\%d.jpg",i);
		tmp_img = cvLoadImage(res,CV_LOAD_IMAGE_GRAYSCALE);
		load_data(T,tmp_img,i);
	}
	
	//��T�����е�ƽ��ֵ
	calc_mean(T,m);

	//����Э�������
	calc_covariance_matrix(T,L,m);

	//��L������ֵ����������
	iteration = 60;
	cstrq(L,TRAIN_NUM,q,b,c);
	flag = csstq(TRAIN_NUM,b,c,q,eps,iteration); //����q�е�j��Ϊ����b�е�j������ֵ��Ӧ����������
	if (flag<0)
	{
		printf("fucking failed!\n");
	}else
	{
		printf("success to get eigen value and vector\n");
	}

	//��L��ѡ���ʵ�����ֵ��������������
	num_q=0;
	for (i=0;i<TRAIN_NUM;i++)
	{
		if (b[i]>1)
		{
			num_q++;
		}
	}
	p_q = (double *)malloc(sizeof(double)*TRAIN_NUM*TRAIN_NUM);			//��ѡ���L�����������������ˣ�δ����
	projected_train = (double *)malloc(sizeof(double)*TRAIN_NUM*num_q);	//ͶӰ���ѵ�����������ռ�
	eigenvector = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*num_q);//Pe=��e,Q(Xe)=��(Xe)��ͶӰ�任����
	pick_eignevalue(b,q,p_q,num_q);
	get_eigenface(p_q,T,num_q,projected_train,eigenvector);

	//��ȡ����ͼ��
	test_img = cvLoadImage(".\\TestDatabase\\4.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	projected_test = (double *)malloc(sizeof(double)*num_q*1);//�������ռ�ͶӰ��Ĳ�������
	for (i=0;i<IMG_HEIGHT;i++)
	{
		for (j=0;j<IMG_WIDTH;j++)
		{
			T_test[i*IMG_WIDTH+j] = (double)(unsigned char)test_img->imageData[i*IMG_WIDTH+j] - m[i*IMG_WIDTH+j];
		}
	}

	//����������ͶӰ�������ռ�
	memset(projected_test,0,sizeof(double)*num_q);
	matrix_mutil(projected_test,eigenvector,T_test,num_q,IMG_WIDTH*IMG_HEIGHT,1);

	//����projected_test��projected_train��ÿ��������ŷ�Ͼ���
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
	//Ѱ����С����
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












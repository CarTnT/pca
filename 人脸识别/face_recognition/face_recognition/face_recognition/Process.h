#include "cv.h"
#include "highgui.h"
#define TRAIN_NUM 20
#define IMG_HEIGHT 200
#define IMG_WIDTH 180

void load_data(double *T,IplImage *src,int k);
void calc_mean(double *T,double *m);
void calc_covariance_matrix(double *T,double *L,double *m);
void pick_eignevalue(double *b,double *q,double *p_q,int num_q);
void get_eigenface(double *p_q,double *T,int num_q,double *projected,double *eigenvector);
#include <math.h>
#include <stdio.h>

void cstrq(double a[],int n,double q[],double b[],double c[]);
int csstq(int n,double b[],double c[],double q[],double eps,int l);
void matrix_mutil(double *c,double *a,double *b,int x,int y,int z);
void matrix_reverse(double *src,double *dest,int row,int col);
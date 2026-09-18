/* Model-free regression: a prefill batch may wrap the sliding KV ring. */
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while(0)
static void check_ring(uint32_t n, uint32_t pos0, uint32_t cap) {
    const uint32_t width=128;
    const size_t values=(size_t)n*width, cached=(size_t)cap*width;
    float *input=malloc(values*sizeof(float));
    _Float16 *actual=malloc(cached*sizeof(_Float16));
    CHECK(input && actual);
    ds4_gpu_tensor *q=ds4_gpu_tensor_alloc(values*4), *k=ds4_gpu_tensor_alloc(values*4),
        *v=ds4_gpu_tensor_alloc(values*4), *gate=ds4_gpu_tensor_alloc(n*4),
        *heads=ds4_gpu_tensor_alloc(values*4), *sk=ds4_gpu_tensor_alloc(values*2),
        *sv=ds4_gpu_tensor_alloc(values*2), *kc=ds4_gpu_tensor_alloc(cached*2),
        *vc=ds4_gpu_tensor_alloc(cached*2);
    CHECK(q&&k&&v&&gate&&heads&&sk&&sv&&kc&&vc);
    for(size_t i=0;i<values;i++)input[i]=0;
    CHECK(ds4_gpu_tensor_write(q,0,input,values*4));
    CHECK(ds4_gpu_tensor_write(gate,0,input,n*4));
    for(size_t i=0;i<cached;i++)actual[i]=(_Float16)-1;
    CHECK(ds4_gpu_tensor_write(kc,0,actual,cached*2));
    CHECK(ds4_gpu_tensor_write(vc,0,actual,cached*2));
    for(size_t i=0;i<values;i++)input[i]=(float)(i/width+1);
    CHECK(ds4_gpu_tensor_write(k,0,input,values*4));
    for(size_t i=0;i<values;i++)input[i]=-(float)(i/width+1);
    CHECK(ds4_gpu_tensor_write(v,0,input,values*4));
    CHECK(ds4_gpu_laguna_attention_prefill_tensor(heads,kc,vc,sk,sv,q,k,v,gate,pos0,n,cap,1,1,width,0.125f));
    for(int which=0;which<2;which++) {
        CHECK(ds4_gpu_tensor_read(which?vc:kc,0,actual,cached*2));
        for(uint32_t row=0;row<cap;row++) {
            int last=-1;
            for(uint32_t t=0;t<n;t++)if((pos0+t)%cap==row)last=(int)t;
            _Float16 want=last<0?(_Float16)-1:(_Float16)((which?-1:1)*(last+1));
            for(uint32_t col=0;col<width;col++)if(actual[(size_t)row*width+col]!=want) {
                fprintf(stderr,"FAIL n=%u pos=%u cap=%u cache=%d row=%u col=%u got=%g want=%g\n",n,pos0,cap,which,row,col,(double)actual[(size_t)row*width+col],(double)want);exit(1);
            }
        }
    }
    ds4_gpu_tensor *all[]={q,k,v,gate,heads,sk,sv,kc,vc};
    for(size_t i=0;i<sizeof(all)/sizeof(all[0]);i++)ds4_gpu_tensor_free(all[i]);
    free(input);free(actual);
}
int main(void) {
    CHECK(ds4_gpu_init());
    for(int repeat=0;repeat<8;repeat++) {
        check_ring(551,0,512); check_ring(1025,37,512);
        check_ring(511,7,512); check_ring(512,0,512);
        check_ring(513,0,512); check_ring(551,0,1024);
    }
    ds4_gpu_cleanup();puts("PASS: prefill KV ring retains newest rows");return 0;
}

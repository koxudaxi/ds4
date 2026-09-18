#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "ds4_gpu.h"
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static void read_value(void *model, uint64_t size, uint64_t offset, float want) {
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(float));
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc(sizeof(float));
    float input = 2, output = 0;
    CHECK(x && y);
    CHECK(ds4_gpu_tensor_write(x, 0, &input, sizeof(input)));
    CHECK(ds4_gpu_matmul_f32_tensor(y, model, size, offset, 1, 1, x, 1));
    CHECK(ds4_gpu_tensor_read(y, 0, &output, sizeof(output)));
    CHECK(output == want);
    ds4_gpu_tensor_free(y); ds4_gpu_tensor_free(x);
}
int main(void) {
    @autoreleasepool {
        const uint64_t size = 64ull << 20, page = getpagesize();
        float *model = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        CHECK(model != MAP_FAILED && ds4_gpu_init());
        ds4_gpu_set_ssd_streaming(true);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        model[0] = 3; model[(size-page)/sizeof(float)] = 7;
        uint64_t offset = 0, bytes = size;
        CHECK(ds4_gpu_set_model_map_spans(model, size, &offset, &bytes, 1, sizeof(float)));
        uint64_t broad = device.currentAllocatedSize;
        bytes = page;
        CHECK(ds4_gpu_set_model_map_spans(model, size, &offset, &bytes, 1, sizeof(float)));
        CHECK(device.currentAllocatedSize < broad / 2);
        read_value(model, size, 0, 6);
        for (int i = 0; i < 3; i++) {
            CHECK(ds4_gpu_set_model_map_spans(model, size, &offset, &bytes, 1, sizeof(float)));
            read_value(model, size, 0, 6);
        }
        float *aux = mmap(NULL, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        CHECK(aux != MAP_FAILED);
        aux[0] = 11;
        uint64_t aux_offset = 0;
        CHECK(ds4_gpu_set_model_map_spans(aux, page, &aux_offset, &page, 1, sizeof(float)));
        offset = size-page;
        CHECK(ds4_gpu_set_model_map_spans(model, size, &offset, &bytes, 1, sizeof(float)));
        read_value(model, size, offset, 14);
        read_value(aux, page, 0, 22);
        offset = size;
        CHECK(!ds4_gpu_set_model_map_spans(model, size, &offset, &bytes, 1, sizeof(float)));
        offset = 0;
        CHECK(ds4_gpu_set_model_map_spans(model, size, &offset, &bytes, 1, sizeof(float)));
        read_value(model, size, 0, 6);
        /* Resident callers may continue reusing a broader mapping. */
        ds4_gpu_set_ssd_streaming(false);
        bytes = size;
        CHECK(ds4_gpu_set_model_map_spans(model, size, &offset, &bytes, 1, sizeof(float)));
        bytes = page;
        CHECK(ds4_gpu_set_model_map_spans(model, size, &offset, &bytes, 1, sizeof(float)));
        read_value(model, size, size-page, 14);
        ds4_gpu_cleanup(); munmap(model, size); munmap(aux, page);
        puts("PASS: streaming view narrowing and replacement");
    }
    return 0;
}

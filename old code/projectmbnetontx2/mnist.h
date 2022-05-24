#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INSIZE 28
//#include "mbnet.h"

typedef struct mnist_data
{
	double data[INSIZE][INSIZE];
	unsigned int label;
} mnist_data;

static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set, unsigned int *count);

static unsigned int mnist_bin_to_int(char *tmp)
{
	int i;
	unsigned int ret = 0;
	for (i = 0; i < 4; i++)
	{
		ret <<= 8;
		ret |= (unsigned char)tmp[i];
	}
	return ret;
}

static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set, unsigned int *count)
{

	int i;
	char tmp[4];
	unsigned int image_cnt, label_cnt;

	// 1. image and label files open
	FILE *ifp = fopen(image_filename, "rb");
	FILE *lfp = fopen(label_filename, "rb");

	// 2. mnist file formats checking
	// 2-1. the magic numbers of image and label files
	fread(tmp, 1, 4, ifp);
	if (mnist_bin_to_int(tmp) != 2051)
	{
		printf("Not a valid image file\n");
		return -2;
	}

	fread(tmp, 1, 4, lfp);
	if (mnist_bin_to_int(tmp) != 2049)
	{
		printf("Not a valid label file\n");
		return -3;
	}

	// 2-2. numbers of images and labels
	fread(tmp, 1, 4, ifp);
	image_cnt = mnist_bin_to_int(tmp);

	fread(tmp, 1, 4, lfp);
	label_cnt = mnist_bin_to_int(tmp);

	// 2-3. check whether the same number or not
	if (image_cnt != label_cnt)
	{
		printf("The number of images and labels is not same !!!\n");
		return -4;
	}

	unsigned int image_dim[2];
	// 2-4. check the number of rows and columns
	for (i = 0; i < 2; i++)
	{
		fread(tmp, 1, 4, ifp);
		image_dim[i] = mnist_bin_to_int(tmp);
	}

	if (image_dim[0] != 28 || image_dim[1] != 28)
	{
		printf("The number of rows and columns are not 28 !!!\n");
		return -5;
	}

	*count = image_cnt;
	*data_set = (mnist_data *)malloc(sizeof(mnist_data) * image_cnt);

	// int one = 1, zero = 0;
	//  3. load image data as double type (the results from 0.0 to 1.0 dividing unsigned char values by 255.0)
	for (i = 0; i < image_cnt; i++)
	{
		// for(i=0; i<10; i++){
		int j;
		unsigned char read_image[INSIZE * INSIZE];
		mnist_data *d = &(*data_set)[i];
		fread(read_image, 1, INSIZE * INSIZE, ifp);

		for (j = 0; j < INSIZE * INSIZE; j++)
		{
			d->data[j / INSIZE][j % INSIZE] = read_image[j] / 255.0;
			/* // for debugging
			if(j%INSIZE == 0)
				printf("\n");
			else
				printf("%d ", (d->data[j/INSIZE][j%INSIZE] > 0)? one : zero);
			*/
		}

		fread(tmp, 1, 1, lfp);
		d->label = tmp[0];
		// printf("\n=> %d \n", d->label);
	}

	// 4. close opened files
	fclose(ifp);
	fclose(lfp);
	return 0;
}

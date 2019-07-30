#include <iostream>
#include <array>
#include <fstream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_new.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

using host_buffer=thrust::host_vector<float>;
using device_buffer=thrust::device_vector<float>;
using host_mask=thrust::host_vector<int>;
using device_mask=thrust::device_vector<int>;
constexpr int args_padded_size=0;

namespace kernel
{

__device__
int to_1d(const uint2 &index, const int width)
{
	return index.y*width+index.x;
}

__device__
int to_1d(const int x, int y, const int width)
{
	return y*width+x;
}

__device__
void block_solver(float *buffer, float *delta, int *mask, int *args, const int offset)
{
	const int padded_size=args[args_padded_size];
	const int local_size=blockDim.y;
	
	const int threads_per_block=blockDim.x*blockDim.y;
	const int thread_id=threads_per_block*(gridDim.x*blockIdx.y+blockIdx.x)+blockDim.x*threadIdx.y+threadIdx.x;

	const uint2 left_top_global=make_uint2(local_size*blockIdx.x, local_size*blockIdx.y);
	const uint2 global_base=make_uint2(left_top_global.x+1+2*threadIdx.x, left_top_global.y+1+threadIdx.y);
	const uint2 global=make_uint2(global_base.x+offset, global_base.y);

	printf("at block(%2d, %2d) thread(%2d, %2d, id=%4d) target(%2d, %2d, 1d=%4d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, thread_id, global.x, global.y, to_1d(global, padded_size));

	buffer[to_1d(global, padded_size)]=(thread_id+1);
}

__global__
void even_solver(float *buffer, float *delta, int *mask, int *args)
{
	block_solver(buffer, delta, mask, args, threadIdx.y%2);
}

__global__
void odd_solver(float *buffer, float *delta, int *mask, int *args)
{
	block_solver(buffer, delta, mask, args, 1-threadIdx.y%2);
}

}


template <class T>
auto append_padding(const thrust::host_vector<T> &buffer, const T &val, const int original_size)
{
	const int padded_size=original_size+2;
	thrust::host_vector<T> padded_buffer(padded_size*padded_size, val);

	for(int y=1; y<=original_size; ++y)
	{
		for(int x=1; x<=original_size; ++x)
		{
			padded_buffer[y*padded_size+x]=buffer[(y-1)*original_size+x-1];
		}
	}

	return padded_buffer;
}

template <class T>
auto remove_padding(const thrust::host_vector<T> &padded_buffer, const int original_size)
{
	const int padded_size=original_size+2;
	std::vector<T> buffer(original_size*original_size);
	for(int y=1; y<=original_size; ++y)
	{
		for(int x=1; x<=original_size; ++x)
		{
			buffer[(y-1)*original_size+x-1]=padded_buffer[y*padded_size+x];
		}
	}

	return buffer;
}

__host__
auto new_solver(const host_buffer &h_buffer, const host_mask &h_mask, const int global_size, const int local_size)
{
	const int padded_size=global_size+2;
	host_mask h_args;
	h_args.push_back(padded_size);
	device_mask d_args=h_args;

	device_buffer d_buffer=append_padding<float>(h_buffer, 0, global_size);
	device_mask d_mask=append_padding<int>(h_mask, 0, global_size);

	device_buffer d_delta(padded_size*padded_size, 0.f);

	float max_delta=0;
	const float min_delta=std::pow(10, -6);

	const dim3 grid(global_size/local_size, global_size/local_size);
	const dim3 block(local_size/2, local_size);

	try
	{
		do
		{
			kernel::odd_solver<<<grid, block>>>(
				thrust::raw_pointer_cast(d_buffer.data()), 
				thrust::raw_pointer_cast(d_delta.data()),
				thrust::raw_pointer_cast(d_mask.data()),
				thrust::raw_pointer_cast(d_args.data())
			);

			cudaError_t err=cudaDeviceSynchronize();
			if(err!=cudaSuccess)
			{
				std::cout<<cudaGetErrorString(err)<<std::endl;
				break;
			}

			max_delta=*thrust::max_element(thrust::device, d_delta.begin(), d_delta.end());
			std::cout<<"max_delta="<<max_delta<<std::endl;
		}
		while(min_delta<max_delta);
	}
	catch(thrust::system_error &e)
	{
		std::cerr<<"Exception:\n"<<e.what()<<std::endl;
	}

	host_buffer new_h_buffer=d_buffer;
	return remove_padding<float>(new_h_buffer, global_size);
}



class cpu_potential_solver
{
public:
	cpu_potential_solver(const int h)
		: h_(h), buffer_(h, std::vector<float>(h, 0)), h_buffer_(h *h, 0), circles_({circle(0.25*h, 0.75*h, 100, 0.125*h), circle(0.875*h, 0.125*h, 20, 0.05*h)}), h_mask_(h *h, 1)
	{
		for(int y=0; y<h; ++y)
		{
			for(int x=0; x<h; ++x)
			{
				if(x==0 || x==h-1 || y==0 || y==h-1)
				{
					this->h_mask_[y*this->h_+x]=0;
					continue;
				}

				for(const auto &c:this->circles_)
				{
					if(c.includes(x, y))
					{
						this->h_buffer_[y*this->h_+x]=c.v_;
						this->buffer_[y][x]=c.v_;
						this->h_mask_[y*this->h_+x]=0;
						break;
					}
				}
			}
		}
	}

	auto solve()
	{
		//return kernel::solver(this->buffer_, this->host_even_mask, this->host_odd_mask, this->host_global_mask);
		return new_solver(this->h_buffer_, this->h_mask_, this->h_, 8);
	}


private:
	const int h_;
	std::vector<std::vector<float>> buffer_;
	host_buffer h_buffer_;
	host_mask h_mask_;

	class circle
	{
	public:
		circle(const float x, const float y, const int v, const float radius) noexcept
			: x_(x), y_(y), v_(v), radius_(radius)
		{}

		bool includes(const int x, const int y) const noexcept
		{
			return std::pow(x-this->x_, 2)+std::pow(y-this->y_, 2)<=std::pow(this->radius_, 2);
		}

		const float x_, y_;
		const int v_;
		const float radius_;
	};

	std::array<circle, 2> circles_;
};

int main()
{
	constexpr int h=32;
	auto solver=cpu_potential_solver(h);

	const auto result=solver.solve();

	std::ofstream os("./out.csv");

	for(int y=h-1; 0<=y; --y)
	{
		for(int x=0; x<h; ++x)
		{
			os<<result[y*h+x];

			if(x<h-1)
			{
				os<<",";
			}
		}
		os<<std::endl;
	}

	return 0;
}
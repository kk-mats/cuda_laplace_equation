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
constexpr int local_size=32;

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
	const int global_1d=to_1d(make_uint2(local_size*blockIdx.x+1+2*threadIdx.x+offset, local_size*blockIdx.y+1+threadIdx.y), padded_size);
	
	const float prev=buffer[global_1d];
	buffer[global_1d]=(1-mask[global_1d])*prev+mask[global_1d]*(
		buffer[global_1d-padded_size]+buffer[global_1d-1]+
		buffer[global_1d+padded_size]+buffer[global_1d+1]
		)/4;
	delta[global_1d]=fabsf(buffer[global_1d]-prev)/100;
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
auto new_solver(const host_buffer &h_buffer, const host_mask &h_mask, const int global_size)
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

	int ctr=0;
	try
	{
		do
		{
			kernel::even_solver<<<grid, block>>>(
				thrust::raw_pointer_cast(d_buffer.data()),
				thrust::raw_pointer_cast(d_delta.data()),
				thrust::raw_pointer_cast(d_mask.data()),
				thrust::raw_pointer_cast(d_args.data())
				);

			kernel::odd_solver<<<grid, block>>>(
				thrust::raw_pointer_cast(d_buffer.data()), 
				thrust::raw_pointer_cast(d_delta.data()),
				thrust::raw_pointer_cast(d_mask.data()),
				thrust::raw_pointer_cast(d_args.data())
			);
			max_delta=*thrust::max_element(thrust::device, d_delta.begin(), d_delta.end());

			if(ctr%100==0)
			{
				std::cout<<"loop:"<<ctr<<" max_delta="<<max_delta<<std::endl;
			}
			++ctr;
		}
		while(min_delta<max_delta);
	}
	catch(thrust::system_error &e)
	{
		std::cerr<<"Exception:\n"<<e.what()<<std::endl;
	}

	host_buffer new_h_buffer=d_buffer;
	return new_h_buffer;
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
		return new_solver(this->h_buffer_, this->h_mask_, this->h_);
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
	constexpr int h=512;
	auto solver=cpu_potential_solver(h);

	const auto result=solver.solve();

	std::ofstream os("./out.csv");

	for(int y=h; 1<=y; --y)
	{
		for(int x=1; x<=h; ++x)
		{
			os<<result[y*(h+2)+x];

			if(x<h)
			{
				os<<",";
			}
		}
		os<<std::endl;
	}

	return 0;
}
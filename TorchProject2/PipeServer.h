#pragma once

#include <windows.h> 
#include <stdio.h>
#include <conio.h>
#include <tchar.h>
//#include <ATen/ATen.h>

#define BUFSIZE 4096

class PipeServer {
	public:
		PipeServer();

		bool connect();

		bool sendData(float* startPtr, int size);

		int recieveData(float* buf, int size);

	private:
		HANDLE mPipe = INVALID_HANDLE_VALUE;
		float* mMessage = NULL;
		float  mBuf[BUFSIZE];
		BOOL   mConnected = FALSE;
		BOOL   mSuccess = FALSE;
		DWORD  mRead, mToWrite, mWritten, mMode;
		LPTSTR mPipename = TEXT("\\\\.\\pipe\\Foo");
};
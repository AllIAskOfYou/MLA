#include "PipeServer.h"

PipeServer::PipeServer() {}

bool PipeServer::connect(std::string pipename) {
    std::string pipePath = mBasePath + pipename;
    _tcscpy_s(mPipename, pipePath.length() + 1, pipePath.c_str());
	_tprintf(TEXT("\nPipe Server: Main thread awaiting client connection on %s\n"), mPipename);
	mPipe = CreateNamedPipe(
		mPipename,				  // pipe name 
		PIPE_ACCESS_DUPLEX,       // read/write access 
		PIPE_TYPE_MESSAGE |       // message type pipe 
		PIPE_READMODE_MESSAGE |   // message-read mode 
		PIPE_WAIT,                // blocking mode 
		PIPE_UNLIMITED_INSTANCES, // max. instances  
		BUFSIZE,                  // output buffer size 
		BUFSIZE,                  // input buffer size 
		0,                        // client time-out 
		NULL);                    // default security attribute 

	if (mPipe == INVALID_HANDLE_VALUE)
	{
		_tprintf(TEXT("CreateNamedPipe failed, GLE=%d.\n"), GetLastError());
		return false;
	}

	// Wait for the client to connect; if it succeeds, 
	// the function returns a nonzero value. If the function
	// returns zero, GetLastError returns ERROR_PIPE_CONNECTED. 

	mConnected = ConnectNamedPipe(mPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
	return mConnected;
}

bool PipeServer::sendData(float* startPtr, int size) {
    mMode = PIPE_READMODE_MESSAGE;
    mSuccess = SetNamedPipeHandleState(
        mPipe,    // pipe handle 
        &mMode,  // new pipe mode 
        NULL,     // don't set maximum bytes 
        NULL);    // don't set maximum time 
    if (!mSuccess)
    {
        _tprintf(TEXT("SetNamedPipeHandleState failed. GLE=%d\n"), (int)GetLastError());
        return false;
    }

    // Send a message to the pipe server.

    mToWrite = size * sizeof(float);
    //_tprintf( TEXT("Sending %d byte message: \"%s\"\n"), mToWrite, mMessage);

    mSuccess = WriteFile(
        mPipe,                      // pipe handle 
        startPtr,   // message 
        mToWrite,                   // message length 
        &mWritten,                  // bytes written 
        NULL);                      // not overlapped 

    if (!mSuccess)
    {
        _tprintf(TEXT("WriteFile to pipe failed. GLE=%d\n"), (int)GetLastError());
        return false;
    }
    //_tprintf(TEXT("Send: %d bytes.\n"), (int)mWritten);
        return true;
}

int PipeServer::recieveData(float* buf, int size) {
    do
    {
        // Read from the pipe. 

        mSuccess = ReadFile(
            mPipe,    // pipe handle 
            buf,    // buffer to receive reply 
            size * sizeof(float),  // size of buffer 
            &mRead,  // number of bytes read 
            NULL);    // not overlapped 

        if (!mSuccess && GetLastError() != ERROR_MORE_DATA) break;

        //_tprintf(TEXT("\"%s\" %d\n"), mBuf, (int)mRead);
    } while (!mSuccess);  // repeat loop if ERROR_MORE_DATA 

    if (!mSuccess)
    {
        _tprintf(TEXT("ReadFile from pipe failed. GLE=%d\n"), (int)GetLastError());
        return -1;
    }

    return mRead / sizeof(float);
}
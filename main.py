from src.server import Server

if __name__ == "__main__":
    server = Server("localhost", 12345)
    server.start()
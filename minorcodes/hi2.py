import multiprocessing as mp
import time

def f():
    print("Zzz")
    time.sleep(5)

if __name__ == "__main__":
    p = mp.Pool(6)

    _ = [p.apply_async(f) for i in range(12)]
    
    print("end!")
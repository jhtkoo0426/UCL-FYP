# from digit_interface import Digit
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt


DIGIT_SERIAL = "D20607"
OUTPUT_FILE = 'data.npy'
OUTPUT_TEST_FILE = 'data_test.npy'
# matplotlib.use('Qt5Agg')


# def get_readings(d, count):
#     readings = []

#     while len(readings) < count:  
#         time.sleep(2)
#         frame = d.get_frame()
#         print(f"Frame dimensions: {frame.shape}")
#         print(f"reading count: {len(readings)}")
#         readings.append(frame)
    
#     print("Completed getting readings")
#     return readings

# def save_readings(output_file, readings):
#     np.save(output_file, readings)
#     print("Readings as ndarray saved to data-collected.txt :)")

def test_reconstruct_readings(file):
    # Reconstruct the sliced data as we know the shape of each reading (320x240x3)
    reading_data = np.load(file)
    print(reading_data.shape)
    print(f"Shape of first entry of reconstructed data: {reading_data[0].shape}")
    print(reading_data[0][0][0])
    print(reading_data[1][0][0])
    print(reading_data[2][0][0])

    # Choose random images to plot
    rand_indices = np.random.choice(len(reading_data), size=10)

    plt.figure()

    rows=2
    cols=5
    
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1)
        plt.imshow(reading_data[rand_indices[i]])
        plt.axis('off')
    plt.show()



# d = Digit(DIGIT_SERIAL)
# d.connect()
# # d.show_view()
# readings = get_readings(d, 3)
# save_readings(OUTPUT_FILE, readings)
# d.disconnect()


test_reconstruct_readings(OUTPUT_FILE)
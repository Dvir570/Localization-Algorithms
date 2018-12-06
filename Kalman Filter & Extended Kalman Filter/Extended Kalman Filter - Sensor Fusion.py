import sys
import math
import numpy as np
import pandas as pd
from numpy.linalg import inv


def predict():
    # Predict Step
    global x, P
    x = np.matmul(A, x)
    At = np.transpose(A)
    P = np.add(np.matmul(A, np.matmul(P, At)), Q)


def update_lidar(z):
    global x, P
    # Measurement update step
    Y = np.subtract(z, np.matmul(H, x))
    Ht = np.transpose(H)
    S = np.add(np.matmul(H, np.matmul(P, Ht)), R_lidar)
    K = np.matmul(P, Ht)
    Si = inv(S)
    K = np.matmul(K, Si)

    # New state
    x = np.add(x, np.matmul(K, Y))
    P = np.matmul(np.subtract(I, np.matmul(K, H)), P)


def update_radar(z):
    global x, P
    # Measurement update step
    Y = np.subtract(z, cartesian_to_polar())
    Hj = calculate_jacobian_matrix()
    Hjt = np.transpose(Hj)
    S = np.add(np.matmul(Hj, np.matmul(P, Hjt)), R_radar)
    K = np.matmul(P, Hjt)
    try:
        Si = inv(S)
    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        pass
    else:
        K = np.matmul(K, Si)

        # New state
        x = np.add(x, np.matmul(K, Y))
        P = np.matmul(np.subtract(I, np.matmul(K, Hj)), P)


def cartesian_to_polar():
    """
    This is a function that specifies the mapping between our predicted values in
    Cartesian coordinates and Polar coordinates.
    This mapping is required because we are predicting in Cartesian coordinates
    but our measurement (z) that is coming from the RADAR sensor is in Polar Coordinates.
    """
    global x
    px = x[0][0]
    py = x[1][0]
    vx = x[2][0]
    vy = x[3][0]
    rho = math.sqrt(px**2 + py**2)
    if math.fabs(rho) > 0.0001:
        phi = math.atan2(py, px)
        rho_dot = ((px * vx + py * vy) / rho)
    else:
        phi = 0
        rho_dot = 0
    return np.array([[rho],
                     [phi],
                     [rho_dot]])


def calculate_jacobian_matrix():
    """
    This is a function that specifies the mapping between our predicted values in
    Cartesian coordinates and Polar coordinates.
    This mapping is required because we are predicting in Cartesian coordinates
    but our measurement (z) that is coming from the RADAR sensor is in Polar Coordinates.
    """
    global x
    px = x[0][0]
    py = x[1][0]
    vx = x[2][0]
    vy = x[3][0]

    c = px**2 + py**2
    sqrt_c = math.sqrt(c)
    sqrt_c3 = c * sqrt_c

    return np.array([[px/sqrt_c, py/sqrt_c, 0, 0],
                     [-py/c, px/c, 0, 0],
                     [py*(vx*py-vy*px)/sqrt_c3, px*(vy*px-vx*py)/sqrt_c3, px/sqrt_c, py/sqrt_c]])


def calculate_rmse(estimations):
    global ground_truth
    if sys.getsizeof(estimations) != sys.getsizeof(ground_truth) or sys.getsizeof(estimations) == 0:
        print('Invalid estimation or ground_truth data')
    rmse = np.zeros([4, 1])  # Root Mean Square Error
    rmse[0][0] = np.sqrt(((estimations[0][0] - ground_truth[0][0]) ** 2).mean())
    rmse[1][0] = np.sqrt(((estimations[1][0] - ground_truth[1][0]) ** 2).mean())
    rmse[2][0] = np.sqrt(((estimations[2][0] - ground_truth[2][0]) ** 2).mean())
    rmse[3][0] = np.sqrt(((estimations[3][0] - ground_truth[3][0]) ** 2).mean())
    print("rmse = ", rmse)


# Read Input File
measurements = pd.read_csv('obj_pose-laser-radar-synthetic-input.txt', header=None, delim_whitespace=True, skiprows=1)

# Manualy copy initial readings from first row of input file.
prv_time = 1477010443000000/1000000.0
x = np.array([
        [0.312242],
        [0.5803398],
        [0],
        [0]
        ])

# Initialize variables to store ground truth and RMSE values
ground_truth = np.zeros([4, 1])

# Initialize matrices P, A, H, I, Z and R
P = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1000., 0.],
        [0., 0., 0., 1000.]
        ])
A = np.array([
        [1.0, 0, 1.0, 0],
        [0, 1.0, 0, 1.0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0]
        ])
H = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0]
        ])

I = np.identity(4)

# Lidar Cov matrix
R_lidar = np.array([
        [0.0225, 0],
        [0, 0.0225]
        ])

# Radar Cov matrix
R_radar = np.array([
        [0.09, 0, 0],
        [0, 0.0009, 0],
        [0, 0, 0.09]
        ])

noise_ax = 9
noise_ay = 9
Q = np.zeros([4, 4])

# Begin iterating through sensor data
for i in range(len(measurements)):
    new_measurement = measurements.iloc[i, :].values
    # Calculate Timestamp and its power variables
    cur_time = new_measurement[3] / 1000000.0
    dt = cur_time - prv_time
    dt_2 = dt * dt
    dt_3 = dt_2 * dt
    dt_4 = dt_3 * dt
    prv_time = cur_time
    # Updating matrix A with dt value
    A[0][2] = dt
    A[1][3] = dt
    # Updating Q matrix
    Q[0][0] = dt_4 / 4 * noise_ax
    Q[0][2] = dt_3 / 2 * noise_ax
    Q[1][1] = dt_4 / 4 * noise_ay
    Q[1][3] = dt_3 / 2 * noise_ay
    Q[2][0] = dt_3 / 2 * noise_ax
    Q[2][2] = dt_2 * noise_ax
    Q[3][1] = dt_3 / 2 * noise_ay
    Q[3][3] = dt_2 * noise_ay
    # Make prediction
    predict()
    if new_measurement[0] == 'L':
        # Updating sensor readings
        z_lidar = np.zeros([2, 1])
        z_lidar[0][0] = new_measurement[1]  # px measured
        z_lidar[1][0] = new_measurement[2]  # py measured
        # Collecting ground truths
        ground_truth[0] = new_measurement[4]
        ground_truth[1] = new_measurement[5]
        ground_truth[2] = new_measurement[6]
        ground_truth[3] = new_measurement[7]
        update_lidar(z_lidar)
    elif new_measurement[0] == 'R':
        # Updating sensor readings
        z_radar = np.zeros([3, 1])
        z_radar[0][0] = new_measurement[1]  # rho measured
        z_radar[1][0] = new_measurement[2]  # phi measured
        z_radar[2][0] = new_measurement[3]  # rhodot measured
        # Collecting ground truths
        ground_truth[0] = new_measurement[5]
        ground_truth[1] = new_measurement[6]
        ground_truth[2] = new_measurement[7]
        ground_truth[3] = new_measurement[8]
        update_radar(z_radar)
    print('iteration', i, 'x: ', x)
    calculate_rmse(x)

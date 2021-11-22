import numpy as np 
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm

############### Creating a nice image of the Mandelbrot set
"""
The code to produce the image is not our original work but a code found on online.
link url: https://levelup.gitconnected.com/mandelbrot-set-with-python-983e9fc47f56   
"""

# this bit is used to find if a complex number diverges under max_steps and if it does then when
def get_iter(c:complex, thresh:int =4, max_steps:int =25) -> int:
    # Z_(n) = (Z_(n-1))^2 + c
    # Z_(0) = c
    z=c
    i=1
    while i<max_steps and (z*z.conjugate()).real<thresh:
        z=z*z +c
        i+=1
    return i

def plotter(n, thresh, max_steps=25):
    mx = 2.48 / (n-1)
    my = 2.26 / (n-1)
    mapper = lambda x,y: (mx*x - 2, my*y - 1.13)
    img=np.full((n,n), 255)
    for x in range(n):
        for y in range(n):
            it = get_iter(complex(*mapper(x,y)), thresh=thresh, max_steps=max_steps)
            img[y][x] = 255 - it
    return img

n = 1000
img = plotter(n, thresh = 4, max_steps = 50)
plt.imshow(img, cmap = 'plasma')
plt.axis('off')
plt.savefig('mandelbrot.png')
plt.show()


############### Monte Carlo Integration

steps = np.array((50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)) # 10000              # number of iterations per points
number_points = 10000 #np.array((50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000)) # number of sampled points for pure random and latin hypercube samplong
ortho_number_points = 2500 #np.array((100, 900, 2500, 4900, 10000))                     # number of sampled points for orthogonal sampling
N = 25          # number of experiments
boot = 10       # number of bootstrapping subsamples

####### Creating sample


# Orthogonal sampling

def orthogonal_sampling(ns):
    """
    This code is not our original work but a code found on stackexchange.
    link url: https://codereview.stackexchange.com/questions/207610/orthogonal-sampling
    
    """
    assert(np.sqrt(ns) % 1 == 0), "Please insert an even number of samples"
    n = int(np.sqrt(ns))
    # Making a datastructure of a dict with coordinate tuples of a bigger grid with subcoordinate of sub-grid points
    blocks = {(i,j):[(a,b) for a in range(n) for b in range(n)] for i in range(n) for j in range(n)}
    points = [] 
    append = points.append 
    for block in blocks.keys():
        point = random.choice(blocks[block])
        lst_row = [(k1, b) for (k1, b), v in blocks.items() if k1 == block[0]]
        lst_col = [(a, k1) for (a, k1), v in blocks.items() if k1 == block[1]]
        
        for col in lst_col:
            blocks[col] = [a for a in blocks[col] if a[1] != point[1]]
            
        for row in lst_row:
            blocks[row] = [a for a in blocks[row] if a[0] != point[0]]
            # Adjust the points to fit the grid they fall in  
        point = (point[0] + n * block[0], point[1] + n * block[1])
        append(point)
    return points

def RescaleToMandelBrotBounds(samples, lb_x, ub_x, lb_y, ub_y, n):
    """
    This function rescales the range of the generated orthogonal samples to that
    of the mandelbrot set bounds
    """
    x_values = []
    y_values = []
    for sample in samples:
        x_values.append(lb_x + sample[0]*(ub_x-lb_x)/n)
        y_values.append(lb_y + sample[1]*(ub_y-lb_y)/n)

    draws = np.stack((x_values,y_values), axis=-1)

    return draws

# Latin hypercube sampling
"""
The code to produce the image is not our original work but a code found on online.
link url:  https://www.youtube.com/watch?v=r6rp-Qxc9xI   
"""

def latin_draws(n,lb,ub):
    range_= np.linspace(lb, ub, n+1)
    lower_limits = range_[:-1]
    upper_limits = range_[1:]
    points = np.random.uniform(low=lower_limits, high=upper_limits, size=[1,n])
    return points

def latin_hypercube(n, lb_x, ub_x, lb_y, ub_y):
    x_draws = latin_draws(n,lb_x,ub_x)
    y_draws = latin_draws(n, lb_y, ub_y)
    draws = np.stack((x_draws,y_draws), axis=-1)
    np.random.shuffle(draws[0][:,1])
    return draws[0]

# Pure random sampling

def pure_random(n):
    x_draws = np.random.uniform(-2, 0.47, size=[1,n])
    y_draws = np.random.uniform(-1.12, 1.12, size=[1,n])
    draws = np.stack((x_draws,y_draws), axis=-1)
    return draws[0]

# Bootstrapping

def bootstrap(n, boot):
    total_sample = RescaleToMandelBrotBounds(orthogonal_sampling(n), -2, 0.47, -1.12, 1.12, n)
    total_x_sample = total_sample[:,0]
    total_y_sample = total_sample[:,1]
    dict = {}

    for subsample in range(boot):
        x_subsample = np.random.choice(total_x_sample, n // boot)
        y_subsample = np.random.choice(total_y_sample, n // boot)
        dict[subsample] = np.stack((x_subsample,y_subsample), axis=-1)

    return dict


####### Integration, repeating experiments, and plotting

def mandelbrot(data, steps, number_points, sampling_method, boot):              # this function performs one experiments for the Monte Carlo integration
    if sampling_method == 'bootstrap':
        output = 0 
        for array in data.values():
            count = 0
            for point in array:
                z = complex(point[0], point[1])
                c = z
                i=1
                while i < steps and np.absolute(z.real) <= 2:
                    z = z * z +c
                    i += 1

                if np.absolute(z.real) <= 2:
                    count +=  1
            
            area_size = (0.47 + 2) * (1.12 + 1.12)
            fraction_in  = count / (number_points / boot)
            output += fraction_in * area_size / boot 
        
    else:
        count = 0
        for point in data:
            z = complex(point[0], point[1])
            c = z
            i=1
            while i < steps and np.absolute(z.real) <= 2:
                z = z * z +c
                i += 1

            if np.absolute(z.real) <= 2:
                count +=  1
    
        area_size = (0.47 + 2) * (1.12 + 1.12)
        fraction_in  = count / number_points
        output = fraction_in * area_size

    return output

def repeated_int(num_iter, number_points, N, sampling_method, boot):  # this function repeats the Monte Carlo integration for multiple experiments
    sizes_dict = {}
    count = 0

    if type(num_iter) == int:                                       # here, we vary the number of sampled points
        ave_sizes = np.empty(len(number_points))
        for iterations in tqdm(number_points):
            sizes_dict[iterations] = np.empty(N)
            ave_size = 0 

            for experiment in range(N):                             
                if sampling_method == 'bootstrap':
                    result = mandelbrot(bootstrap(iterations, boot), num_iter, iterations, sampling_method, boot)
                elif sampling_method == 'pure':
                    result = mandelbrot(pure_random(iterations), num_iter, iterations, sampling_method, boot)
                elif sampling_method == 'lhs':
                    result = mandelbrot(latin_hypercube(iterations, -2, 0.47, -1.12, 1.12), num_iter, iterations, sampling_method, boot)
                else:
                    result = mandelbrot(RescaleToMandelBrotBounds(orthogonal_sampling(iterations), -2, 0.47, -1.12, 1.12, iterations), num_iter, iterations, sampling_method, boot)
                
                sizes_dict[iterations][experiment] = result
                ave_size += result / N 

            ave_sizes[count] = ave_size
            count += 1
        
    else:                                                           # here, we vary the number of iterations per sampled point
        ave_sizes = np.empty(len(num_iter))
        for iterations in tqdm(num_iter):
            sizes_dict[iterations] = np.empty(N)
            ave_size = 0 

            for experiment in range(N):
                if sampling_method == 'bootstrap':
                    result = mandelbrot(bootstrap(number_points, boot), iterations, number_points, sampling_method, boot)               
                elif sampling_method == 'pure':
                    result = mandelbrot(pure_random(number_points), iterations, number_points, sampling_method, boot)
                elif sampling_method == 'lhs':
                    result = mandelbrot(latin_hypercube(number_points, -2, 0.47, -1.12, 1.12), iterations, number_points, sampling_method, boot)
                else:
                    result = mandelbrot(RescaleToMandelBrotBounds(orthogonal_sampling(number_points), -2, 0.47, -1.12, 1.12, number_points), iterations, number_points, sampling_method, boot)
                
                sizes_dict[iterations][experiment] = result
                ave_size += result / N 
        
            ave_sizes[count] = ave_size
            count += 1

    return ave_sizes, sizes_dict

def plot(steps, area_dict, number_points, N, sampling_method):                  # create a plot of convergence of the area estimate
    if type(steps) == int:
        length = len(number_points)
    else: 
        length = len(steps)

    means = np.empty(length)
    uppers = np.empty(length)
    lowers = np.empty(length)
    count = 0 

    for array in area_dict.items():
        mean = np.mean(array[1])
        stdev = np.std(array[1])
        interval = 1.96 * stdev / (N ** (1/2))

        means[count] = mean
        uppers[count] = mean + interval
        lowers[count] = mean - interval

        count += 1
    
    x = np.linspace(0, length, length)
    plt.plot(x, means, 'blue', label = 'Mean')
    plt.plot(x, lowers, 'red', label = 'Upper bound')
    plt.plot(x, uppers, 'red', label = 'Lower bound')
    plt.plot((x[0], x[-1]), (1.506, 1.506), 'k-')
    
    plt.fill_between(x, lowers, uppers, color = 'lightcoral')
    plt.ylabel("Area of the Mandelbrot set")
    plt.grid()
    plt.legend()
    if type(steps) == int:
        plt.xticks(x, number_points, rotation = 45)
        plt.xlabel("Number of points evaluated")
        plt.title(f"Area of the Mandelbrot set for different numbers points evaluated, \n {N} experiments, 95% CI in bounds.")
        plt.savefig(f"final_images/areas_per_number_points_{N}_{sampling_method}.png")
    else:
        plt.xticks(x, steps, rotation = 45)
        plt.xlabel("Number of iterations per point")
        plt.title(f"Area of the Mandelbrot set for different numbers of iteration per point, \n {N} experiments, 95% CI in bounds.")
        plt.savefig(f"final_images/areas_per_steps_{N}_{sampling_method}.png")
    plt.close()

def diff_plot(steps, area_dict, number_points, N, sampling_method):                 # create plot of the improvement in the area estimate
    if type(steps) == int:
        length = len(number_points) - 1
    else: 
        length = len(steps) - 1

    means = np.empty(length)
    uppers = np.empty(length)
    lowers = np.empty(length)
    count = 0 

    key_less_dict = []
    for array in area_dict.items():
        key_less_dict.append(array)
    
    for i in range(0, length):
        array0 = key_less_dict[i][1]
        array1 = key_less_dict[i+1][1]
        diff = array1 - array0
        mean = np.mean(diff)
        stdev = np.std(diff)
        interval = 1.96 * stdev / (N ** (1/2))

        means[count] = mean
        uppers[count] = mean + interval
        lowers[count] = mean - interval

        count += 1
    
    x = np.linspace(0, length, length)
    plt.plot(x, means, 'blue', label = 'Mean')
    plt.plot(x, lowers, 'red', label = 'Upper bound')
    plt.plot(x, uppers, 'red', label = 'Lower bound')
    
    plt.fill_between(x, lowers, uppers, color = 'lightcoral')
    plt.ylabel("Improvement in estimate of Mandelbrot set area")
    plt.grid()
    plt.legend()
    if type(steps) == int:
        plt.xticks(x, number_points[1:], rotation = 45)
        plt.xlabel("Number of points evaluated")
        plt.title(f"Area of the Mandelbrot set for different numbers points evaluated, \n {N} experiments.")
        plt.savefig(f"final_images/Diff_in_areas_per_number_points_{N}_{sampling_method}.png")
    else:
        plt.xticks(x, steps[1:], rotation = 45)
        plt.xlabel("Number of iterations per point")
        plt.title(f"Area of the Mandelbrot set for different numbers of iteration per point, \n {N} experiments.")
        plt.savefig(f"final_images/Diff_in_areas_per_steps_{N}_{sampling_method}.png")
    plt.close()

####### Function calls

print('Pure random sampling')
areas, areas_dict = repeated_int(steps, number_points, N, 'pure', boot)
print(number_points, areas, areas_dict)
plot(steps, areas_dict, number_points, N, 'Pure random sampling')
diff_plot(steps, areas_dict, number_points, N, 'Pure random sampling')
print()
print()

print('Latin hypercube sampling')
areas, areas_dict = repeated_int(steps, number_points, N, 'lhs', boot)
print(number_points, areas, areas_dict)
plot(steps, areas_dict, number_points, N, 'Latin hypercube sampling')
diff_plot(steps, areas_dict, number_points, N, 'Latin hypercube sampling')
print()
print()

print('Orthogonal sampling')
areas, areas_dict = repeated_int(steps, ortho_number_points, N, 'ortho', boot)
print(number_points, areas, areas_dict)
plot(steps, areas_dict, ortho_number_points, N, 'Orthogonal sampling')
diff_plot(steps, areas_dict, ortho_number_points, N, 'Orthogonal sampling')
print()
print()

print('Bootstrapping + orthogonal sampling')
areas, areas_dict = repeated_int(steps, ortho_number_points, N, 'bootstrap', boot)
print(number_points, areas, areas_dict)
plot(steps, areas_dict, ortho_number_points, N, 'Bootstrapping')
diff_plot(steps, areas_dict, ortho_number_points, N, 'Bootstrapping')
print()
print()



###################  Varying both the number of sampled points and the number of iterations per sampled points

        
steps = np.array((50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000))                  # number of iterations per sampled point
number_points = np.array((50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000))          # number of sampled points
N = 50              # number of experiments

def draw():                 # create one random point
    x = np.random.uniform(-2, 0.47)
    y = np.random.uniform(-1.12, 1.12)
    return complex(x, y)

def mandelbrot(c, steps):       # check whether point is in the Mandelbrot set
    z=c
    i=1
    while i < steps and np.absolute(z.real) <= 2:
        z = z * z +c
        i += 1
    if np.absolute(z.real) <= 2:
        return 1
    else:
        return 0

def integrate(steps, number_points):            # monte carlo integration
    count = 0 

    for point in range(number_points):
        count += mandelbrot(draw(), steps)
    
    area_size = (0.47 + 2) * (1.12 + 1.12)
    fraction_in  = count / number_points
    return fraction_in * area_size

def repeated_int(num_iter, number_points, N):           # monte carlo integration for many experiments
    sizes_dict = {}
    ave_sizes = [] #np.empty(len(num_iter))
    count1 = 0

    for iterations in tqdm(num_iter):
        sizes_dict[iterations] = {}
        ave_sizes.append(np.empty(len(number_points)))
        #ave_sizes[count1] = np.empty(len(number_points))
        count2 = 0
        
        for evaluations in number_points:
            sizes_dict[iterations][evaluations] = np.empty(N)
            ave_size = 0 

            for experiment in range(N):
                result = integrate(iterations, evaluations)
                sizes_dict[iterations][evaluations][experiment] = result
                ave_size += result / N 

            ave_sizes[count1][count2] = ave_size 
            count2 += 1

        count1 += 1

    return ave_sizes, sizes_dict

def plot(steps, number_points, ave_sizes, N):                   # create a 3d plot of convergence
    x = np.linspace(0, len(steps), len(steps))
    y = np.linspace(0, len(number_points), len(number_points))[::-1]
    X, Y = np.meshgrid(x, y)
    z = np.array(ave_sizes)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, z, color='blue')
    #ax.plot_surface(X, Y, z, alpha=1)

    xx, yy = np.meshgrid(x, y)
    zz = []
    for element in range(len(steps)):
        zz.append(np.full(len(number_points), 1.506))
    ax.plot_surface(xx, yy, np.array(zz), alpha=0.5)

    plt.xticks(x,steps)
    plt.yticks(y,number_points)
    ax.set_xlabel("Number of iterations per point")
    ax.set_ylabel("Number of points evaluated")
    ax.set_zlabel("Area of the Mandelbrot set")
    # plt.title(f"Area of the Mandelbrot set based on {N} experiments.")
    plt.savefig(f"areas_{N}.png")
    plt.show()


areas, areas_dict = repeated_int(steps, number_points, N)
print(number_points, areas, areas_dict)
plot(steps, number_points, areas, N)
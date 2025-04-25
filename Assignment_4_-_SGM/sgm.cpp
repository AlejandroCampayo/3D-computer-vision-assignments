#include <stdio.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <utility>
using namespace std;


extern "C" {

// Hint: you can use fprintf(stderr, ...) to print messages

//
// The following provide macros to index multidimensional arrays
// Use array[INDEX(...)]
// d stands for disparity
//
#define INDEX2(x, y)                     ( y*w + x )
#define INDEX3(x, y, d)                ( ( y*w + x )*max_d + d )
#define INDEX4(dir, x, y, d) ( ( ( dir*h + y)*w + x )*max_d + d )



// Regularization
double regularization_term(int dp, int dq, double P1, double P2) {
    if (dp == dq) {
        return 0;
    } else if (abs(dp - dq) == 1) {
        return P1;
    } else {
        return P2;
    }
}

// Structure to represent directions
struct Direction {
    int dx;
    int dy;

    Direction(int x, int y) : dx(x), dy(y) {}
};

// Check if a pixel is within valid range
bool isValidCoordinate(int x, int y, int w, int h) {
    return (0 <= x && x < w && 0 <= y && y < h);
}

// Compute min cost
double compute_min_cost(const double* data_term, int nx, int ny, int w, int h, int max_d, int dp, double P1, double P2) {
    double min_cost = INFINITY;
    double cost;
    for (int dq = 0; dq < max_d; ++dq) {
        cost = data_term[INDEX3(nx, ny, dq)] + regularization_term(dp, dq, P1, P2);
        min_cost = min(min_cost, cost);
    }

    return min_cost;
}

void compute_messages(const double* data_term, double* messages, int h, int w, double P1, double P2, int max_d) {
      vector<Direction> directions = {
        {0, 1},  // Right
        {1, 1},  // Down-Right
        {1, 0},  // Down
        {1, -1}, // Down-Left
        {0, -1}, // Left
        {-1, -1},// Up-Left
        {-1, 0}, // Up
        {-1, 1}  // Up-Right
    };

    for (int y = 1; y < h; ++y) {
        for (int x = 1; x < w; ++x) {
          for (int direction = 0; direction < directions.size(); ++direction){
                int dx = directions[direction].dx;
                int dy = directions[direction].dy;
                int nx = x + dx;
                int ny = y + dy;

                if (isValidCoordinate(nx, ny, w, h)) {
                  for (int dp = 0; dp < max_d; ++dp) {
                    double min_cost = compute_min_cost(data_term, nx, ny, w, h, max_d, dp, P1, P2);
                    messages[INDEX4(direction, x, y, dp)] = data_term[INDEX3(x, y, dp)] + min_cost;

                  }
                }
          }
      }
    }
}

void compute_energy(const double* messages, double* E, int h, int w, int max_d) {

    // Traverse through all pixels and disparities
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int d = 0; d < max_d; ++d) {
                double cost = 0.0;

                // Sum up messages from all directions
                for (int direction = 0; direction < 8; ++direction) {
                    cost += messages[INDEX4(direction, x, y, d)];
                }

                // Store the aggregated cost
                E[INDEX3(x, y, d)] = cost;
            }
        }
    }
}

void compute_disparities(const double* E, uint8_t* disparities, int h, int w, int max_d) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double min_cost = INFINITY;
            //disparities[INDEX2(x, y)] = 0;

            for (int d = 0; d < max_d; ++d) {
                if (E[INDEX3(x, y, d)] < min_cost) {
                    min_cost = E[INDEX3(x, y, d)];
                    disparities[INDEX2(x, y)] = static_cast<uint8_t>(d);
                }
            }
        }
    }
}

//
// SGM algorithm in C
//
void sgm(double* data_term, double* messages, uint8_t* disparities, int w, int h, int max_d, float P1, float P2)
{
  // Dimensions:
  //   data_term:   w * h * max_d
  //   messages:    8 * w * h * max_d     [writable]
  //   disparities: w * h                 [writable]

  disparities[0] = 23;
  fprintf(stderr, "Running SGM C code w=%d, h=%d, max_d=%d, P1=%f, P2=%f\n", w, h, max_d, P1, P2);

  // Create a zero message
  double *zero_message = new double[max_d];
  for(int i=0; i < max_d; i++)
    zero_message[i] = 0;

  // *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  // NOTE: It is recommended to introduce functions to simpilfy your code

  // Send messages in all directions
  compute_messages(data_term, messages, h, w, P1, P2, max_d);

  // Compute the aggregated cost for each pixel
  double *E = new double[w * h * max_d];
  compute_energy(messages, E, h, w, max_d);


  // Select disparity according to minimum cost
  compute_disparities(E, disparities, h, w, max_d);

  // *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  // Delete variables
  delete[] zero_message;
}

}

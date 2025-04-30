/* UMANS: Unified Microscopic Agent Navigation Simulator
** MIT License
** Copyright (C) 2018-2020  Inria Rennes Bretagne Atlantique - Rainbow - Julien Pettré
**
** Permission is hereby granted, free of charge, to any person obtaining
** a copy of this software and associated documentation files (the
** "Software"), to deal in the Software without restriction, including
** without limitation the rights to use, copy, modify, merge, publish,
** distribute, sublicense, and/or sell copies of the Software, and to
** permit persons to whom the Software is furnished to do so, subject
** to the following conditions:
**
** The above copyright notice and this permission notice shall be
** included in all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
** OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
** NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
** LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
** ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
** CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**
** Contact: crowd_group@inria.fr
** Website: https://project.inria.fr/crowdscience/
** See the file AUTHORS.md for a list of all contributors.
*/

#include "tools/localsearch.h"
#include "tools/vector2D.h"

#include <iostream>
using namespace std;


// (𝑥 − 2)^2  ==> 𝑥* = 2
double testfunc(Vector2D x, Vector2D* grad = nullptr)
{
    double y = pow(x.x() -2, 2);
    cout << "f(" << x.x() << ") = " << y << endl;
    if (grad)
    {
        double g = 2 * pow((x.x()-2), 1);
        cout << "grad(" << x.x() << ") = " << g << endl;
        grad->set(g, 0);
    }
    return y;
}

double f(Vector2D x)
{
    return testfunc(x, nullptr);
}

int main(int argc, char *argv[])
{
    Vector2D x0(0, 0);
    Vector2D g;

    int N_ITR = 5;
    for (int i=0; i<N_ITR; i++) {
        testfunc(x0, &g);
        double alfa = LocalSearch::backtr(x0, -g, f, 10, 1e-4, 0.5);
        x0 = x0 - alfa * g;
    }
    cout << "********** Result ***************\n" ;
    f(x0);
}

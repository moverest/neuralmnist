// Copyright 2016 Clément Martinez

// Package neuralnetminst provides tools to convert mnist sets to neuralnet
// sets.
package neuralnetminst

import (
	"math"

	"github.com/moverest/mnist"
	"github.com/moverest/neuralnet"
)

const (
	// A represents the activated state
	A = 1
	// D represents the deactivated state
	D = 0
)

// Labels are the neural network vectors associated with each labels
var Labels = [...][]float64{
	[]float64{A, D, D, D, D, D, D, D, D, D},
	[]float64{D, A, D, D, D, D, D, D, D, D},
	[]float64{D, D, A, D, D, D, D, D, D, D},
	[]float64{D, D, D, A, D, D, D, D, D, D},
	[]float64{D, D, D, D, A, D, D, D, D, D},
	[]float64{D, D, D, D, D, A, D, D, D, D},
	[]float64{D, D, D, D, D, D, A, D, D, D},
	[]float64{D, D, D, D, D, D, D, A, D, D},
	[]float64{D, D, D, D, D, D, D, D, A, D},
	[]float64{D, D, D, D, D, D, D, D, D, A},

	// labels which are not in the MNIST database
	[]float64{D, D, D, D, D, D, D, D, D, D},
}

// Set defines a neural network set.
type Set struct {
	In    [][]float64
	Out   [][]float64
	Label []mnist.Label
}

// ConvertImage converts a mnist.Image to a []float64.
func ConvertImage(img *mnist.Image) []float64 {
	vect := make([]float64, len(img))
	for i, v := range img {
		vect[i] = (float64(v)/255.)*math.Abs(A-D) + math.Min(A, D)
	}

	return vect
}

// ConvertLabel converts a mnist.Label to a []float64.
func ConvertLabel(l mnist.Label) []float64 {
	if l < 0 || l > 9 {
		return Labels[10]
	}

	return Labels[l]
}

// ConvertSet converts a mnist.Set to a neuralnet.Set.
func ConvertSet(ms *mnist.Set) *Set {
	n := ms.Count()

	s := &Set{
		In:    make([][]float64, n),
		Out:   make([][]float64, n),
		Label: make([]mnist.Label, n),
	}

	for i := 0; i < n; i++ {
		s.In[i] = ConvertImage(ms.Images[i])
		s.Out[i] = ConvertLabel(ms.Labels[i])
		s.Label[i] = ms.Labels[i]
	}

	return s
}

// Count counts the number of items in the set.
func (s *Set) Count() int {
	return len(s.In)
}

// GetVects return the input and output vectors in the set at the index i.
func (s *Set) GetVects(i int) (in, out []float64) {
	in = s.In[i]
	out = s.Out[i]
	return
}

func maxFloatlSliceValueIndex(s []float64) int {
	if len(s) == 0 {
		return 0
	}

	maxI := 0
	for i := range s {
		if s[i] > s[maxI] {
			maxI = i
		}
	}

	return maxI
}

// Evaluate return the number of successful guesses made by the network.
func Evaluate(net *neuralnet.Network, test Set) int {
	nCorrects := 0
	for i := 0; i < test.Count(); i++ {
		out := net.FeedForward(test.In[i])
		if maxFloatlSliceValueIndex(out) == int(test.Label[i]) {
			nCorrects++
		}
	}

	return nCorrects
}

## `mi_partition.py`

Study the effects of dividing into partitions.

![picture of mutual information to each number of quantiles](mutual_info_partitioning.png "mi partitions")

## Performance measure

Profiling the memory:

```Python
from memory_profiler import profile
from time import sleep

@profile
def carregar_arquivo():
    with open('br-utf8.txt') as file:
        conteudo = file.read()
    return conteudo
```

Checking the performance:

```Python
import cProfile

prof = cProfile.Profile()

prof.enable()
# Do something
prof.disable()
prof.dump_stats('my_profile.prof')
```

Then run some analysis tool:

```
gprof2dot -f pstats my_profile.prof | dot -Tpng -o out.png
```

Measure the performance using the time:

```Python
from timeit import timeit

def test():
    pass

print(
     timeit('test()', globals=globals(), number=10_000)
)
```

## Percentile Test

Output:
```
Method = inverted_cdf time: 0.460187 seconds.
Method = averaged_inverted_cdf time: 0.86104 seconds.
Method = closest_observation time: 0.413436 seconds.
Method = interpolated_inverted_cdf time: 0.847766 seconds.
Method = hazen time: 0.866025 seconds.
Method = weibull time: 0.872946 seconds.
Method = linear time: 0.876766 seconds.
Method = median_unbiased time: 0.865695 seconds.
Method = normal_unbiased time: 0.863017 seconds.
```
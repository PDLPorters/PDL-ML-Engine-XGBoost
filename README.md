# PDL::ML::Engine::XGBoost

XGBoost gradient boosting C API (`xgboost/c_api.h`) bindings for [PDL](https://pdl.perl.org/), enabling fast gradient-boosted decision tree training, inference, and model persistence directly from Perl.

## Prerequisites

- Perl 5.10+
- `PDL` (>= 2.080)
- `libxgboost-dev` (`xgboost/c_api.h`)

## Installation

```bash
perl Makefile.PL
make
make test
make install
```

Or with `cpanm`:

```bash
cpanm .
```

## Quickstart

```perl
use strict;
use warnings;
use PDL;
use PDL::ML::Engine::XGBoost qw(dmatrix train load_model xgboost_version);

my $X = float([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]);
my $y = float([3, 5, 7, 9, 11, 13]);

my $dm = dmatrix($X, $y);
my $booster = train(
    $dm, undef,
    num_boost_round => 60,
    params          => {
        objective => 'reg:squarederror',
        max_depth => 3,
        eta       => 0.3,
    },
);

my $preds = $booster->predict($dm);
$booster->save_model('model.ubj');
my $reloaded = load_model('model.ubj');
```

## Examples & Documentation

- Runnable quickstart: [examples/quickstart.pl](examples/quickstart.pl) (`perl examples/quickstart.pl`)
- Full API documentation and tutorial: `perldoc PDL::ML::Engine::XGBoost`

# Copyright 2026 Google LLC
#
# Licensed under the the same terms as Perl itself (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

use strict;
use warnings;
use File::Temp qw(tempfile);
use PDL;
use PDL::ML::Engine::XGBoost qw(dmatrix train load_model xgboost_version);

print 'XGBoost version: ', xgboost_version(), "\n";

# 1. Build a DMatrix from 2D feature and 1D label PDL ndarrays
my $X = float([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
]);
my $y = float([3, 5, 7, 9, 11, 13]);

my $dm = dmatrix($X, $y);
print 'DMatrix shape: ', $dm->num_row(), ' x ', $dm->num_col(), "\n";

# 2. Train a gradient-boosted decision tree regressor
my $booster = train(
    $dm, undef,
    num_boost_round => 60,
    params          => {
        objective => 'reg:squarederror',
        max_depth => 3,
        eta       => 0.3,
    },
);

# 3. Predict and evaluate residual error
my $preds   = $booster->predict($dm);
my $max_err = abs($preds - $y)->max();
print 'Predictions:   ', $preds, "\n";
print 'Max abs error: ', $max_err, "\n";

# 4. Save model to UBJ and reload
my ($fh, $model_path) = tempfile('pdl_xgb_XXXXXX', SUFFIX => '.ubj', UNLINK => 1);
close $fh;

$booster->save_model($model_path);
my $reloaded = load_model($model_path);
my $preds2   = $reloaded->predict($dm);
print 'Reload max diff: ', abs($preds - $preds2)->max(), "\n";

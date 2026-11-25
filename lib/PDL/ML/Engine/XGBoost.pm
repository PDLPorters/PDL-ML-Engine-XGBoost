package PDL::ML::Engine::XGBoost;

use strict;
use warnings;
use Exporter 'import';
use Carp qw(croak);
use XSLoader;

use PDL::Core;
use PDL::Types;
use PDL::Slices;

our $VERSION = '0.011';

our @EXPORT_OK = qw(
    dmatrix
    train
    load_model
    xgboost_version
);

XSLoader::load('PDL::ML::Engine::XGBoost', $VERSION);

sub dmatrix {
    my ($x, $y, %opts) = @_;
    croak 'dmatrix requires a 2D PDL ndarray' unless defined $x;
    my $missing = exists $opts{missing} ? $opts{missing} : 'NaN' + 0;
    my $xf = $x->float();
    my $dm = PDL::ML::Engine::XGBoost::DMatrix::_xs_from_pdl($xf, $missing);
    if (defined $y) {
        $dm->set_label($y->float());
    }
    return $dm;
}

sub train {
    my ($x, $y, %opts) = @_;
    my $rounds = delete $opts{num_boost_round} || 20;
    my $params = delete $opts{params}          || {};

    my $dm = (ref($x) && $x->isa('PDL::ML::Engine::XGBoost::DMatrix'))
        ? $x
        : dmatrix($x, $y);

    my $booster = PDL::ML::Engine::XGBoost::Booster::_xs_create($dm);
    for my $k (sort keys %{$params}) {
        $booster->set_param($k, '' . $params->{$k});
    }
    for my $i (0 .. $rounds - 1) {
        $booster->update($dm, $i);
    }
    return $booster;
}

sub load_model {
    my ($path) = @_;
    croak 'load_model requires a file path' unless defined $path && length $path;
    my $booster = PDL::ML::Engine::XGBoost::Booster::_xs_create(undef);
    $booster->load_model($path);
    return $booster;
}

1;

__END__

=encoding UTF-8

=head1 NAME

PDL::ML::Engine::XGBoost - XGBoost gradient boosting C API bindings for PDL

=head1 SYNOPSIS

    use PDL;
    use PDL::ML::Engine::XGBoost qw(dmatrix train load_model xgboost_version);

    my $X = float([[1, 2], [2, 3], [3, 4], [4, 5]]);
    my $y = float([3, 5, 7, 9]);

    my $booster = train(
        $X, $y,
        num_boost_round => 50,
        params          => {
            objective => 'reg:squarederror',
            max_depth => 3,
            eta       => 0.2,
        },
    );

    my $preds = $booster->predict(dmatrix($X));
    $booster->save_model('model.ubj');

=head1 DESCRIPTION

C<PDL::ML::Engine::XGBoost> binds L<PDL> ndarrays directly to XGBoost's pure-C API
(C<xgboost/c_api.h>) via C99 XS, enabling fast gradient-boosted decision tree
training, inference, and model persistence directly from Perl.

=head1 FUNCTIONS

=head2 dmatrix

=for ref

Create a C<PDL::ML::Engine::XGBoost::DMatrix> from a 2D L<PDL> feature matrix (and optional
1D label ndarray).

=for usage

    $dm = dmatrix($X);
    $dm = dmatrix($X, $y);

=head2 train

=for ref

Train a C<PDL::ML::Engine::XGBoost::Booster> for C<num_boost_round> iterations with the
given XGBoost parameter hashref.

=for usage

    $booster = train($X, $y, num_boost_round => 50, params => \%params);

=head2 load_model

=for ref

Load a saved XGBoost model (JSON/UBJ) from disk into a C<PDL::ML::Engine::XGBoost::Booster>.

=for usage

    $booster = load_model($path);

=head2 xgboost_version

=for ref

Return the linked C<libxgboost> version string (e.g. C<3.0.4>).

=for usage

    $ver = xgboost_version();

=head1 TUTORIAL

This tutorial adapts the official XGBoost quickstart (constructing a
C<DMatrix>, training a gradient-boosted C<Booster>, predicting, and saving /
reloading the model) to idiomatic Perl and L<PDL>.

=head2 1. Build a DMatrix from PDL ndarrays

Convert a 2D feature matrix C<$X> and 1D label vector C<$y> into an XGBoost
C<DMatrix> using C<dmatrix>:

    use strict;
    use warnings;
    use File::Temp qw(tempfile);
    use PDL;
    use PDL::ML::Engine::XGBoost qw(dmatrix train load_model xgboost_version);

    print 'XGBoost version: ', xgboost_version(), "\n";

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

=head2 2. Train a Booster and Predict

Pass either C<($dm, undef)> or raw C<($X, $y)> to C<train> along with
C<num_boost_round> and a C<params> hashref, then call C<predict> on a
C<DMatrix>:

    my $booster = train(
        $dm, undef,
        num_boost_round => 60,
        params          => {
            objective => 'reg:squarederror',
            max_depth => 3,
            eta       => 0.3,
        },
    );

    my $preds   = $booster->predict($dm);
    my $max_err = abs($preds - $y)->max();
    print 'Predictions:   ', $preds, "\n";
    print 'Max abs error: ', $max_err, "\n";

=head2 3. Save and Reload a Model

Persist the trained booster in Universal Binary JSON (C<.ubj>) or JSON format
with C<save_model> and reload it later with C<load_model>:

    my ($fh, $model_path) = tempfile('pdl_xgb_XXXXXX', SUFFIX => '.ubj', UNLINK => 1);
    close $fh;

    $booster->save_model($model_path);

    my $reloaded = load_model($model_path);
    my $preds2   = $reloaded->predict($dm);
    print 'Reload diff:   ', abs($preds - $preds2)->max(), "\n";

See F<examples/quickstart.pl> in the distribution for a complete runnable
script.

=head1 TYPES

Supports standard real numeric PDL types (C<byte>, C<short>, C<long>, C<longlong>, C<float>, C<double> as supported by the underlying library). Complex types (C<cfloat>, C<cdouble>) are not supported by this backend.

=head1 SEE ALSO

L<PDL>, L<PDL::Torch>, L<PDL::IO::Arrow>, L<PDL::IO::Protobuf>

=head1 AUTHOR

C.J. Collier <cjac@google.com>

=head1 COPYRIGHT AND LICENSE

Copyright 2026 Google LLC

This library is free software; you can redistribute it and/or modify it under
the same terms as Perl itself.

=cut

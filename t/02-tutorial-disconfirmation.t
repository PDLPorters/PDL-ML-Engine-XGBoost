use strict;
use warnings;
use Test::More;
use PDL;
use PDL::ML::Engine::XGBoost;

subtest 'disconfirmation: loading non-existent model file fails cleanly' => sub {
    my $ok = eval {
        my $b = PDL::ML::Engine::XGBoost::Booster->new();
        $b->load_model('/nonexistent/path/to/model.ubj');
        1;
    };
    ok(!$ok, 'load_model dies on nonexistent file');
};

done_testing();

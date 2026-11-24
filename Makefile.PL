use 5.010;
use strict;
use warnings;
use ExtUtils::MakeMaker;
use PDL::Core::Dev;

my $brew_prefix = '';
if ($^O eq 'darwin') {
    chomp($brew_prefix = `brew --prefix 2>/dev/null` || '');
}
my @inc_dirs = ('/usr/include', '/usr/local/include');
push @inc_dirs, "$brew_prefix/include" if $brew_prefix;
my $xgb_inc;
for my $dir (@inc_dirs) {
    if (-f "$dir/xgboost/c_api.h") {
        $xgb_inc = $dir;
        last;
    }
}
if (!$xgb_inc && system('pkg-config --exists xgboost 2>/dev/null') != 0) {
    die "libxgboost-dev (xgboost/c_api.h) is required to build PDL::ML::Engine::XGBoost.\n";
}
my $inc_flag = $xgb_inc ? ' -I' . $xgb_inc : '';
my $lib_flag = ($brew_prefix && $xgb_inc && $xgb_inc =~ m{^\$brew_prefix}) ? "-L$brew_prefix/lib " : '';

my $pdl_inc = PDL::Core::Dev::PDL_INCLUDE();

WriteMakefile(
    NAME               => 'PDL::ML::Engine::XGBoost',
    AUTHOR             => q{C.J. Collier <cjac@google.com>},
    VERSION_FROM       => 'lib/PDL/ML/Engine/XGBoost.pm',
    ABSTRACT_FROM      => 'lib/PDL/ML/Engine/XGBoost.pm',
    LICENSE            => 'apache_2_0',
    MIN_PERL_VERSION   => '5.010',
    CONFIGURE_REQUIRES => {
        'ExtUtils::MakeMaker' => '0',
        'PDL'                 => '2.080',
    },
    TEST_REQUIRES => {
        'Test::More' => '0.88',
    },
    PREREQ_PM => {
        'PDL' => '2.080',
    },
    INC     => '-I. ' . $pdl_inc . $inc_flag,
    XSMULTI => 1,
    LIBS    => [ $lib_flag . '-lxgboost' ],
    clean   => { FILES => 'PDL-ML-Engine-XGBoost-* *.o XGBoost.bs XGBoost.c' },
    META_MERGE => {
        'meta-spec' => { version => 2 },
        resources   => {
            repository => {
                type => 'git',
                url  => 'https://github.com/PDLPorters/PDL-ML-Engine-XGBoost.git',
                web  => 'https://github.com/PDLPorters/PDL-ML-Engine-XGBoost',
            },
            bugtracker => {
                web => 'https://github.com/PDLPorters/PDL-ML-Engine-XGBoost/issues',
            },
        },
    },
);

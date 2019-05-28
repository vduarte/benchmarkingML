

function out=matlab(nB, repeats)
    logy_grid = load('logy_grid.txt');
    Py = load('P.txt');



    beta = .953;
    gamma = 2.;
    r = 0.017;
    theta = 0.282;
    ny = size(logy_grid, 1);

    Bgrid = linspace(-.45, .45, nB);
    ygrid = exp(logy_grid);

    ymean = mean(ygrid);
    def_y = min(0.969 * ymean, ygrid);

    Vd = zeros(ny, 1);
    Vc = zeros(ny, nB);
    V = zeros(ny, nB);
    Q = ones(ny, nB) * .95;

    y = reshape(ygrid, ny, 1, 1);
    B = reshape(Bgrid, 1, nB, 1);
    Bnext = reshape(Bgrid, 1, 1, nB);

    zero_ind = ceil(nB / 2);

    u = @(c) c.^(1 - gamma) / (1 - gamma);

    tic()
    for iteration = 1:repeats
        EV = Py * V;
        EVd = Py * Vd;
        EVc = Py * Vc;

        Vd_target = u(def_y) + beta * (theta * EVc(:, zero_ind) + (1 - theta) * EVd(:, 1));
        Vd_target = reshape(Vd_target, ny, 1);

        Qnext = reshape(Q, ny, 1, nB);

        c = max(y - Qnext .* Bnext + B, 1e-14);
        EV = reshape(EV, ny, 1, nB);
        m = u(c) + beta * EV;
        Vc_target = max(m, [], 3);

        default_states = Vd > Vc;
        default_prob = Py * default_states;
        Q_target = (1 - default_prob) / (1 + r);

        V_upd = max(Vc, Vd);

        V = V_upd;
        Vc = Vc_target;
        Vd = Vd_target;
        Q = Q_target;
    end

    out = toc() / repeats;

end

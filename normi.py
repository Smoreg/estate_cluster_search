# encoding: utf-8
# Good clustering model cian search

import pandas
from sklearn import preprocessing
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN

from pprint import pprint
from sklearn.metrics import adjusted_rand_score

import warnings

warnings.filterwarnings("ignore")


# Winner ---  ['pr_clus30-17-10', 0.88718961785305817]

# ---------------- normalaizers -----------------

# preprocessing.scale
# preprocessing.normalize

def death_link_score(a, b):
    old = 0
    new = 0
    for num, elem in enumerate(a):
        links_old = set([n for n, l_elem in enumerate(a) if l_elem == a[num]])
        links_new = set([n for n, l_elem in enumerate(b) if l_elem == b[num]])
        old += len(links_old - links_new)
        new += len(links_new - links_old)
        # res -= len(links_old.symmetric_difference(links_new))
    return dict(
        links_lost=old / 2,
        links_bad=new / 2,
        clust_diff=abs(len(set(a)) - len(set(b)))
    )


class EstateList:
    def __init__(self, file_path):
        self.data = pandas.read_csv(file_path)
        self.clus_num = 1

    def add_clus_partition(self, grouper, name=None, ):
        if not name:
            name = "clus" + str(self.clus_num)
            self.clus_num += 1

        self.data[name] = grouper(self.data)


class Grouper:
    def __init__(self, norm=lambda x: x):
        self.normer = norm
        self.keys = 'pr la ha'.split()


class PrGrouper(Grouper):
    def __init__(self, pr=25, ha=10, la=10, *norm):
        # self.config = [pr, ha, la]
        self.config = [float(i) / 100 for i in [pr, ha, la]]
        self.config_dict = dict(
            flex=dict(zip(['pr', 'ha', 'la'], self.config)),
            solid={
                "neg": [],
                "pos": []
            }
        )
        super().__init__(*norm)

    class Home:
        cluster = 0

        # price, home, land = config

        def __init__(self, **kwargs):
            self.keys = kwargs.keys()
            for some_field in kwargs:
                setattr(self, some_field, kwargs[some_field])

        def dist(self, other):
            pass

        def __repr__(self):
            res = ""
            for key in self.keys:
                res += str(getattr(self, key)).ljust(15, '-') + ' '
            return res

        def validate(self):
            return all([self.house_area, self.land_area, self.building_type == u'Дом'])

        def get(self, field):
            return getattr(self, field)

    class Cluster:
        class ClusterRange:
            def __init__(self, x, y, conf):
                self.mi = min(x, y)
                self.ma = max(x, y)
                self.conf = conf

            def __contains__(self, item):
                if self.mi <= item <= self.ma:
                    return True
                else:
                    if self.mi > item:
                        return item * (1 + self.conf) >= self.ma
                    elif self.ma < item:
                        return self.mi * (1 + self.conf) >= item
                    else:
                        raise Exception('Some wtf in ckuster_range')

        def __init__(self, cluster_id, base):
            self.base = base
            self.cluster_id = cluster_id
            self.elems = []

        def _add_home(self, obj):
            if not obj.cluster:
                obj.cluster = self.cluster_id
                self.elems.append(obj)
            else:
                raise Exception('Already in cluster')

        def values_list(self, val):
            res = list()
            for elem in self.elems:
                res.append(getattr(elem, val))
            return res

        def _check_containity(self):
            config_dict = self.base.config_dict
            val_fork = dict()
            val_solid = dict(pos={}, neg={})

            for cond in config_dict['flex']:
                vals = self.values_list(cond)
                fmax = max(vals)
                fmin = min(vals)
                val_fork[cond] = self.ClusterRange(fmax, fmin, config_dict['flex'][cond])

            for sign in ['pos', 'neg']:
                for cond in config_dict['solid'][sign]:
                    val_solid[sign][cond] = self.elems[0].get(cond)

            return val_fork, val_solid

        def try_append(self, elem):
            if not self.elems:
                self._add_home(elem)
                return True

            val_fork, val_solid = self._check_containity()
            flag = \
                all([getattr(elem, cond) in fork for cond, fork in val_fork.items()]) \
                and all([getattr(elem, cond) == val if val else True for cond, val in val_solid['pos'].items()]) \
                # and all([getattr(elem, cond) != val for cond, val in val_solid['neg'].items()])

            if flag:
                self._add_home(elem)
                return True
            else:
                return False

        def __repr__(self):

            if not self.elems:
                return "Empty id{}".format(self.cluster_id)
            else:
                res = ''
                for key in self.elems[0].keys:
                    res += str(getattr(self, key)).ljust(15, '-') + ' '
                return [repr(e) for e in self.elems]

    def __call__(self, data):
        # data = self.normer(data.as_matrix(self.keys))
        data = data.T.to_dict().values()
        data = [self.Home(**x) for x in data]
        clusters = []
        for home in data:
            for cluster in clusters:
                if cluster.try_append(home):
                    break
            else:
                new_clus = self.Cluster(len(clusters) + 1, self)
                new_clus.try_append(home)
                clusters.append(new_clus)

        return [x.cluster for x in data]


class EvGrouper(Grouper):
    def __init__(self, dist=0.15, *norm):
        self.dist = dist

        super().__init__(*norm)

    def __call__(self, data):
        data = self.normer(data.as_matrix(self.keys))
        return DBSCAN(eps=self.dist, min_samples=1).fit_predict(data)


class MeanShiftGrouper(Grouper):
    bandwidth = None

    def __init__(self, quantile, n_samples=750, *norm):
        self.quantile = quantile
        self.n_samples = n_samples
        super().__init__(*norm)

    def __call__(self, data):
        data = self.normer(data.as_matrix(self.keys))
        bandwidth = estimate_bandwidth(data, self.quantile, self.n_samples)
        ms = MeanShift(bandwidth)
        ms.fit(data)
        return [int(ms.predict([x])) for x in data]


def clus_pack():
    OPT = True
    sher_li = EstateList("in_data.csv")
    mill_li = EstateList("mil.csv")

    sim_s = preprocessing.scale
    nor_s = preprocessing.normalize
    mm_s = preprocessing.MinMaxScaler().fit_transform
    non_s = lambda x: x

    scalers = [sim_s, nor_s, non_s, mm_s]

    # -------------main

    # li.add_clus_partition(PrGrouper(25, 5, 5, sim_s), 'pr_clus25-5-5')

    # for i in range(100):
    #    li.add_clus_partition(EvGrouper(i / 100, sim_s), 'eu_clus{}'.format(i))
    # li.add_clus_partition(MeanShiftGrouper(6 / 100, 750, sim_s), 'ms_clus6')

    # -------------opt
    if OPT:
        for s_num, scale in enumerate(scalers):
            step = 5
            add = lambda *x: [sher_li.add_clus_partition(*x), mill_li.add_clus_partition(*x)]
            #print(scale, 'pr')
            #Winner ---
            # {'ARI_SHER': 0.84956331647147032,
            # 'ARI': 0.8082358930786202,
            # 'ARI_MILL': 0.76690846968576998,
            # 'Name': 'pr_clus[6, 15, 10, 0]'}
            # if not scalers.index(scale):
            #     for i0 in range(1, 50, step):
            #         for i1 in range(5, 25, step):
            #             for i2 in range(5, 30, step):
            #                 add(
            #                     PrGrouper(i0, i1, i2, scale),
            #                     'pr_clus{}'.format([i0, i1, i2, scalers.index(scale)])
            #                 )

            # print(scale, 'eu')
            # for i in range(1, 400):
            #     try:
            #         add(EvGrouper(i / 1000, scale), 'eu_clus{}'.format([i, scalers.index(scale)]))
            #     except ValueError:
            #         pass
            if s_num == 2:
                print(scale, 'ms')
                for i in range(1, 100, 2):
                    try:
                        add(MeanShiftGrouper(i / 100, 44, sim_s), 'ms_clus{}'.format([i, scalers.index(scale)]))
                    except ValueError:
                        pass
        print('clustering complete')
    # li.data.to_csv("work.csv")

    # --------------validate
    data = mill_li.data
    orig = data['clus']
    keys = [i for i in data.keys() if "_clus" in i]
    scores = []
    for i in keys:
        score = {}
        # score.update(death_link_score(orig, data[i]))
        # score.update(
        #     {'size': len(set(data[i])),
        #      'original_size': len(set(orig)),
        #      })
        score.update(
            {
                'Name': i,
                'ARI_SHER': adjusted_rand_score(sher_li.data['clus'], sher_li.data[i]),
                'ARI_MILL': adjusted_rand_score(mill_li.data['clus'], mill_li.data[i]),

            })
        score.update(
            {
                    'ARI': (score['ARI_SHER'] + score['ARI_MILL'])/2,
            }
        )
        scores.append(score)
    for i in 'pr_ ms_ eu_'.split():
        lscores = list(filter(lambda x: i in x['Name'], scores))
        if lscores:
            print(i, "!" * 50)
            for j in (sorted(lscores, key=lambda x: -x['ARI'])[:10]):
                print(j)
            # print('Winner --- ', max(lscores, key=lambda x: x['ARI']))
            print('Winner SHER--- ', max(lscores, key=lambda x: x['ARI_SHER']))
            print('Winner MILL--- ', max(lscores, key=lambda x: x['ARI_MILL']))

    print('Winner TOTAL--- ', max(scores, key=lambda x: x['ARI']))
    # pprint(scores)


def start():
    clus_pack()

# start()

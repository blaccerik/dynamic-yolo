from sqlalchemy import Integer, Column, BigInteger, ForeignKey, Float

from project import db


class ModelResults(db.Model):
    id = Column(BigInteger, primary_key=True)
    model_id = Column(BigInteger, ForeignKey("model.id"), nullable=False)
    subset_id = Column(Integer, ForeignKey("subset.id"), nullable=False)
    epoch = Column(Integer, nullable=True)
    metric_precision = Column(Float, nullable=False, default=0)
    metric_recall = Column(Float, nullable=False, default=0)
    metric_map_50 = Column(Float, nullable=False, default=0)
    metric_map_50_95 = Column(Float, nullable=False, default=0)
    val_box_loss = Column(Float, nullable=False, default=0)
    val_obj_loss = Column(Float, nullable=False, default=0)
    val_cls_loss = Column(Float, nullable=False, default=0)

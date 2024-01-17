import pandas as pd


class Preprocess:
    def __init__(self, df):
        self.df = df
        self.data = self.load_data(df)

    def load_data(self, df):
        self.df = df
        subj_entity, subj_type = [], []
        obj_entity, obj_type = [], []
        sub_idx, obj_idx = [], []
        sentences = []

        for i in range(len(df)):
            subj_dict = eval(df["subject_entity"][i])
            obj_dict = eval(df["object_entity"][i])
            subj_entity.append(subj_dict["word"])
            subj_type.append(subj_dict["type"])
            sub_idx.append((subj_dict["start_idx"], subj_dict["end_idx"]))
            obj_entity.append(obj_dict["word"])
            obj_type.append(obj_dict["type"])
            obj_idx.append((obj_dict["start_idx"], obj_dict["end_idx"]))
            sentences.append(df["sentence"][i])

        entity_df = pd.DataFrame(
            {"subject_entity": subj_entity, "subject_type": subj_type, "object_entity": obj_entity, "object_type": obj_type, "subject_idx": sub_idx, "object_idx": obj_idx, "sentence": sentences}
        )

        return entity_df

    def entity_marker(self, data, df):
        data = self.data
        sents = []

        for i in range(len(data)):
            sent = data.sentence[i]
            subj_i = data.subject_idx[i]
            obj_i = data.object_idx[i]
            if subj_i[0] < obj_i[0]:
                sent = sent[: subj_i[0]] + " @ " + sent[subj_i[0] : subj_i[1] + 1] + " @ " + sent[subj_i[1] + 1 : obj_i[0]] + " # " + sent[obj_i[0] : obj_i[1] + 1] + " # " + sent[obj_i[1] + 1 :]
            else:
                sent = sent[: obj_i[0]] + " # " + sent[obj_i[0] : obj_i[1] + 1] + " # " + sent[obj_i[1] + 1 : subj_i[0]] + " @ " + sent[subj_i[0] : subj_i[1] + 1] + " @ " + sent[subj_i[1] + 1 :]
            sents.append(sent)

        return pd.DataFrame({"sentence": sents, "subject_entity": data.subject_entity, "object_entity": data.object_entity, "label": data.label, "source": data.source})

    def typed_entity_marker(self, df):
        data = self.load_data(df)
        sents = []

        for i in range(len(data)):
            sent = data.sentence[i]
            subj_t = data.subject_type[i]
            obj_t = data.object_type[i]
            subj_i = data.subject_idx[i]
            obj_i = data.object_idx[i]
            if subj_i[0] < obj_i[0]:
                sent = (
                    sent[: subj_i[0]]
                    + " @ * "
                    + subj_t
                    + " * "
                    + sent[subj_i[0] : subj_i[1] + 1]
                    + " @ "
                    + sent[subj_i[1] + 1 : obj_i[0]]
                    + " # * "
                    + obj_t
                    + " * "
                    + sent[obj_i[0] : obj_i[1] + 1]
                    + " # "
                    + sent[obj_i[1] + 1 :]
                )
            else:
                sent = (
                    sent[: obj_i[0]]
                    + " # * "
                    + obj_t
                    + " * "
                    + sent[obj_i[0] : obj_i[1] + 1]
                    + " # "
                    + sent[obj_i[1] + 1 : subj_i[0]]
                    + " @ * "
                    + subj_t
                    + " * "
                    + sent[subj_i[0] : subj_i[1] + 1]
                    + " @ "
                    + sent[subj_i[1] + 1 :]
                )

            sents.append(sent)

        return pd.DataFrame({"sentence": sents, "subject_entity": df.subject_entity, "object_entity": df.object_entity, "label": df.label, "source": df.source})

    def add_qa_relation(self, df):
        data = self.load_data(df)
        qa_relations = []

        for i in range(len(data)):
            subj_entity = data.subject_entity[i]
            obj_entity = data.object_entity[i]

            subject_type = data.subject_type[i]
            object_type = data.object_type[i]

            relation_qa = f" @ * {subject_type} * {subj_entity} @ 와(과)  # * {object_type} * {obj_entity} # 의 관계는 무엇인가?"
            qa_relations.append(relation_qa)

        df["qa_relation"] = qa_relations
        return df
